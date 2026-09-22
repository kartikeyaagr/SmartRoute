"""
Evaluation datasets, one per routing path.

The router has three destinations, and a dataset that exercises only one of them
cannot tell you whether routing works. MMLU — the only dataset this repo had — is
500 single-question, 4-way multiple-choice items; 3 of 500 have any multi-part
structure, so it cannot evaluate the decomposition layer at all.

    dataset    what it looks like                     should route to
    ---------  -------------------------------------  ---------------
    TriviaQA   "Who was the man behind The Chipmunks?" CHEAP     (lookup)
    MMLU       domain question, 4 options, one letter  MIDDLE    (atomic)
    HotpotQA   multi-hop, needs 2+ facts combined      DECOMPOSE (layer 2)

Dataset membership *is* the ground-truth routing label, which makes it both the
evaluation target and the training set for the classifier heads — no hand-labelling.

Corpora are materialised to JSONL under harness/data/ (see build_datasets.py) so runs
are deterministic and offline, matching the existing mmlu_500.jsonl.
"""

from __future__ import annotations

import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DATA_DIR = Path(__file__).parent / "data"

# The routing destinations a dataset can expect. Matches catalog tier aliases plus
# the pseudo-destination for "needs layer 2".
LOOKUP = "cheap"
ATOMIC = "middle"
DECOMPOSE = "decompose"


@dataclass
class Example:
    source_id: str
    prompt: str
    answer: str
    dataset: str
    expected_path: str
    aliases: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class Dataset(Protocol):
    name: str
    expected_path: str

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]: ...

    def grade(self, example: Example, response: str) -> tuple[bool, bool]:
        """Returns (is_correct, extraction_failed)."""
        ...

    def system_prompt(self) -> str: ...


# ---------------------------------------------------------------------------
# Grading helpers
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles, extra space."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(g) for g in golds if g)


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 — the standard partial-credit metric for short-answer QA."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, golds: list[str]) -> bool:
    """
    Substring containment on normalised text.

    Free-text models rarely emit a bare entity — "The man behind The Chipmunks was
    Ross Bagdasarian, aka David Seville" is correct but fails exact match. Graded
    generously on purpose: this benchmark compares *models against each other* on the
    same grader, so a uniformly lenient metric still ranks them correctly, whereas a
    strict one mostly measures instruction-following.
    """
    pred = normalize_answer(prediction)
    return any(normalize_answer(g) in pred for g in golds if g)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Generate it:  uv run harness/data/synthetic_corpus.py"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _sample(rows: list, n: int | None, seed: int) -> list:
    if n is None or n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SyntheticDataset:
    """
    The 200-question corpus written for this project (harness/data/synthetic_corpus.py).

    Replaces the academic sets as the primary benchmark. MMLU is 4-way multiple choice,
    TriviaQA is deliberately obscure, and HotpotQA's 2-hop questions are synthetically
    welded together — none of it resembles what an assistant is actually asked, so
    routing tuned on that traffic is tuned on the wrong distribution.

    Graded on `must_include`: a list of requirements, each a string that must appear or
    a list of acceptable alternatives. Deterministic, free, and no LLM judge.

    Sub-sets are exposed by category so the per-tier priors can be read off per kind of
    request rather than averaged into a single meaningless number.
    """

    name = "synthetic"
    expected_path = ATOMIC  # per-item; this is only the registry default

    def __init__(self, path: Path | None = None, category: str | None = None) -> None:
        self._path = path or DATA_DIR / "synthetic_200.jsonl"
        self._category = category

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]:
        rows = _read_jsonl(self._path)
        if self._category:
            rows = [r for r in rows if r["category"] == self._category]
        rows = _sample(rows, n, seed)
        return [
            Example(
                source_id=r["source_id"],
                prompt=r["prompt"],
                answer="",  # graded by assertion, not by a single gold string
                dataset=self.name,
                expected_path=r["expected_path"],
                metadata={"category": r["category"], "must_include": r["must_include"]},
            )
            for r in rows
        ]

    def system_prompt(self) -> str:
        # No format coercion beyond brevity: the point is to measure models on the
        # register a user actually writes in, not on instruction-following.
        return "Answer the question directly and concisely."

    def grade(self, example: Example, response: str) -> tuple[bool, bool]:
        if not response.strip():
            return False, True
        text = normalize_answer(response)
        for requirement in example.metadata.get("must_include", []):
            alternatives = [requirement] if isinstance(requirement, str) else requirement
            if not any(normalize_answer(a) in text for a in alternatives if a):
                return False, False
        return True, False


# ---------------------------------------------------------------------------
# Registry and the mixed stream
# ---------------------------------------------------------------------------

DATASETS: dict[str, type] = {
    SyntheticDataset.name: SyntheticDataset,
}

DEFAULT_DATASETS = [SyntheticDataset.name]


def get_dataset(name: str) -> Dataset:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}. Known: {sorted(DATASETS)}")
    return DATASETS[name]()


def mixed_stream(
    per_dataset: int = 100,
    seed: int = 42,
    names: list[str] | None = None,
) -> list[tuple[Example, Dataset]]:
    """
    An interleaved, shuffled stream across all three datasets.

    This is the real test of the classifier: given traffic it cannot tell apart by
    provenance, does each item still reach the tier its dataset implies?
    """
    pairs: list[tuple[Example, Dataset]] = []
    for name in names or DEFAULT_DATASETS:
        dataset = get_dataset(name)
        pairs.extend((ex, dataset) for ex in dataset.load(per_dataset, seed=seed))
    random.Random(seed).shuffle(pairs)
    return pairs
