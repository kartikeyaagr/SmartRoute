"""Tests for the evaluation corpus and its grader."""

import pytest

from harness.corpora import SyntheticDataset, normalize_answer


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  The Chipmunks, Inc.! ", "chipmunks inc"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("David   Seville", "david seville"),
        ("", ""),
    ],
)
def test_normalize(raw, expected):
    assert normalize_answer(raw) == expected


def test_corpus_labels_all_three_paths_per_item():
    """One corpus carries every destination, which is what lets it both train and evaluate."""
    examples = SyntheticDataset().load()
    assert len(examples) == 200
    assert {e.expected_path for e in examples} == {"cheap", "middle", "decompose"}


def test_grading_rejects_wrong_and_evasive_answers():
    """Keyword grading is only useful if it actually says no."""
    ds = SyntheticDataset()
    item = next(e for e in ds.load() if "capital of Australia" in e.prompt)
    assert ds.grade(item, "Canberra is the capital.")[0]
    assert not ds.grade(item, "The capital of Australia is Sydney.")[0]
    assert ds.grade(item, "   ") == (False, True)


def test_every_item_has_assertions():
    """An item with no assertions is unconditionally correct and inflates accuracy."""
    assert all(e.metadata["must_include"] for e in SyntheticDataset().load())
