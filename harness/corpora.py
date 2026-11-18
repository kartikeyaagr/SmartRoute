"""
The evaluation corpus: 200 questions written for this project
(harness/data/synthetic_corpus.py), each labelled with the routing path it should take.

    expected_path  what it looks like
    -------------  -----------------------------------------
    cheap          a lookup someone types instead of searching
    middle         ordinary atomic reasoning or writing
    decompose      multi-part work that layer 2 should split

The per-item label is both the evaluation target and the training set for the triage
heads, so no hand-labelling is needed. MMLU/TriviaQA/HotpotQA were dropped: none of
them resembles what an assistant is actually asked, so routing tuned on them is tuned
on the wrong distribution.
"""

from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCTUATION = str.maketrans("", "", string.punctuation)


@dataclass
class Example:
    source_id: str
    prompt: str
    dataset: str
    expected_path: str
    metadata: dict = field(default_factory=dict)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles, extra space."""
    return " ".join(_ARTICLES.sub(" ", text.lower().translate(_PUNCTUATION)).split())


class SyntheticDataset:
    """
    Graded on `must_include`: a list of requirements, each a string that must appear or
    a list of acceptable alternatives. Deterministic, free, and no LLM judge.
    """

    name = "synthetic"

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DATA_DIR / "synthetic_200.jsonl"

    def load(self, n: int | None = None, seed: int = 42) -> list[Example]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Generate it:  uv run harness/data/synthetic_corpus.py"
            )
        with self.path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if n is not None and n < len(rows):
            rows = random.Random(seed).sample(rows, n)
        return [
            Example(
                source_id=r["source_id"],
                prompt=r["prompt"],
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
        """Returns (is_correct, extraction_failed)."""
        if not response.strip():
            return False, True
        text = normalize_answer(response)
        for requirement in example.metadata.get("must_include", []):
            alternatives = [requirement] if isinstance(requirement, str) else requirement
            if not any(normalize_answer(a) in text for a in alternatives if a):
                return False, False
        return True, False
