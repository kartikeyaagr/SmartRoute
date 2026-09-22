"""Tests for evaluation corpora and their graders."""

import json

import pytest

from harness.corpora import (
    ATOMIC,
    DATASETS,
    DECOMPOSE,
    LOOKUP,
    Example,
    SyntheticDataset,
    contains_answer,
    exact_match,
    get_dataset,
    mixed_stream,
    normalize_answer,
    token_f1,
)


class TestNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  The Chipmunks, Inc.! ", "chipmunks inc"),
            ("A Tale of Two Cities", "tale of two cities"),
            ("David   Seville", "david seville"),
            ("", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_answer(raw) == expected


class TestGraders:
    def test_exact_match_ignores_case_and_punctuation(self):
        assert exact_match("David Seville.", ["david seville"])

    def test_exact_match_rejects_different_answer(self):
        assert not exact_match("Alvin", ["David Seville"])

    def test_contains_answer_handles_conversational_responses(self):
        """Free-text models wrap the fact in a sentence; that is still correct."""
        response = "The man behind The Chipmunks was Ross Bagdasarian, also known as David Seville."
        assert contains_answer(response, ["David Seville"])

    def test_contains_answer_rejects_absent_fact(self):
        assert not contains_answer("I am not sure who that was.", ["David Seville"])

    @pytest.mark.parametrize(
        "pred,gold,floor",
        [("Barack Obama", "Barack Obama", 1.0), ("Barack Hussein Obama", "Barack Obama", 0.7)],
    )
    def test_token_f1_partial_credit(self, pred, gold, floor):
        assert token_f1(pred, gold) >= floor

    def test_token_f1_zero_on_disjoint(self):
        assert token_f1("completely different", "Barack Obama") == 0.0


class TestRegistry:
    def test_synthetic_is_the_default_and_labels_per_item(self):
        """
        The synthetic corpus carries all three destinations in one file, labelled per
        question — which is what lets one corpus both train and evaluate the router.
        """
        from harness.corpora import DEFAULT_DATASETS

        assert DEFAULT_DATASETS == ["synthetic"]
        assert list(DATASETS) == ["synthetic"]
        examples = SyntheticDataset().load()
        assert len(examples) == 200
        assert {e.expected_path for e in examples} == {LOOKUP, ATOMIC, DECOMPOSE}

    def test_synthetic_grading_rejects_wrong_and_evasive_answers(self):
        """Keyword grading is only useful if it actually says no."""
        ds = SyntheticDataset()
        item = next(e for e in ds.load() if "capital of Australia" in e.prompt)
        assert ds.grade(item, "Canberra is the capital.")[0]
        assert not ds.grade(item, "The capital of Australia is Sydney.")[0]
        assert ds.grade(item, "   ") == (False, True)

    def test_synthetic_every_item_has_assertions(self):
        """An item with no assertions is unconditionally correct and inflates accuracy."""
        assert all(e.metadata["must_include"] for e in SyntheticDataset().load())

    def test_get_dataset_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            get_dataset("nope")

    def test_every_dataset_has_a_system_prompt(self):
        assert all(get_dataset(n).system_prompt() for n in DATASETS)
