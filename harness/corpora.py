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
            f"{path} not found. Build it first:  uv run harness/build_corpora.py"
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

class MMLUDataset:
    """Single-step, knowledge-heavy, 4-way multiple choice. Should route to MIDDLE."""

    name = "mmlu"
    expected_path = ATOMIC
    VALID_ANSWERS = frozenset("ABCD")

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DATA_DIR / "mmlu_500.jsonl"

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]:
        rows = _sample(_read_jsonl(self._path), n, seed)
        return [
            Example(
                source_id=r["source_id"],
                prompt=r["prompt"],
                answer=r["correct_answer"],
                dataset=self.name,
                expected_path=self.expected_path,
                metadata={"subject": r.get("subject"), "label_tier": r.get("difficulty_tier")},
            )
            for r in rows
        ]

    def system_prompt(self) -> str:
        return "Respond with ONLY the letter A, B, C, or D on the first line."

    def grade(self, example: Example, response: str) -> tuple[bool, bool]:
        extracted, failed = self.extract_letter(response)
        return (extracted == example.answer and not failed), failed

    @staticmethod
    def extract_letter(response: str) -> tuple[str, bool]:
        """Extract A/B/C/D from the first line. Returns (letter, extraction_failed)."""
        first_line = response.strip().split("\n")[0].strip().upper()
        for token in first_line.split():
            clean = token.strip("().:,*")
            if clean in MMLUDataset.VALID_ANSWERS:
                return clean, False
        return (first_line[:1] if first_line else ""), True


class TriviaQADataset:
    """
    The google-substitute tier: short single-hop factual lookups.

    This is the traffic layer 1 should divert to the cheapest model — questions
    someone asks instead of typing them into a search box.
    """

    name = "triviaqa"
    expected_path = LOOKUP

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DATA_DIR / "triviaqa_500.jsonl"

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]:
        rows = _sample(_read_jsonl(self._path), n, seed)
        return [
            Example(
                source_id=r["source_id"],
                prompt=r["prompt"],
                answer=r["answer"],
                aliases=tuple(r.get("aliases") or ()),
                dataset=self.name,
                expected_path=self.expected_path,
            )
            for r in rows
        ]

    def system_prompt(self) -> str:
        return "Answer with just the fact, as briefly as possible. No explanation."

    def grade(self, example: Example, response: str) -> tuple[bool, bool]:
        golds = [example.answer, *example.aliases]
        if not response.strip():
            return False, True
        return (exact_match(response, golds) or contains_answer(response, golds)), False


class HotpotQADataset:
    """
    Multi-hop questions that genuinely need two or more facts combined.

    This is the only dataset here that can exercise layer 2 — the decomposition
    path — because it is the only one whose items have more than one part.
    """

    name = "hotpotqa"
    expected_path = DECOMPOSE

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DATA_DIR / "hotpotqa_500.jsonl"

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]:
        rows = _sample(_read_jsonl(self._path), n, seed)
        return [
            Example(
                source_id=r["source_id"],
                prompt=r["prompt"],
                answer=r["answer"],
                dataset=self.name,
                expected_path=self.expected_path,
                metadata={"type": r.get("type"), "level": r.get("level")},
            )
            for r in rows
        ]

    def system_prompt(self) -> str:
        return "Answer with just the fact, as briefly as possible. No explanation."

    def grade(self, example: Example, response: str) -> tuple[bool, bool]:
        golds = [example.answer]
        if not response.strip():
            return False, True
        if exact_match(response, golds) or contains_answer(response, golds):
            return True, False
        # yes/no comparison items must match exactly; partial credit would inflate them
        if example.answer.lower() in ("yes", "no"):
            return False, False
        return token_f1(response, example.answer) >= 0.6, False


# ---------------------------------------------------------------------------
# Registry and the mixed stream
# ---------------------------------------------------------------------------

DATASETS: dict[str, type] = {
    MMLUDataset.name: MMLUDataset,
    TriviaQADataset.name: TriviaQADataset,
    HotpotQADataset.name: HotpotQADataset,
}


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
    for name in names or list(DATASETS):
        dataset = get_dataset(name)
        pairs.extend((ex, dataset) for ex in dataset.load(per_dataset, seed=seed))
    random.Random(seed).shuffle(pairs)
    return pairs
