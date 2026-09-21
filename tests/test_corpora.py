"""Tests for evaluation corpora and their graders."""

import json

import pytest

from harness.corpora import (
    ATOMIC,
    DECOMPOSE,
    LOOKUP,
    DATASETS,
    Example,
    HotpotQADataset,
    MMLUDataset,
    TriviaQADataset,
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


class TestMMLU:
    @pytest.mark.parametrize(
        "response,letter,failed",
        [
            ("B", "B", False),
            ("B. Because the rate is...", "B", False),
            ("(C)", "C", False),
            ("The answer is D", "D", False),
            ("I am not certain about this", "I", True),
            ("", "", True),
        ],
    )
    def test_extract_letter(self, response, letter, failed):
        assert MMLUDataset.extract_letter(response) == (letter, failed)

    def test_grades_against_gold_letter(self):
        ds = MMLUDataset()
        ex = Example("x", "q", "B", "mmlu", ATOMIC)
        assert ds.grade(ex, "B. because") == (True, False)
        assert ds.grade(ex, "A") == (False, False)

    def test_extraction_failure_is_not_correct(self):
        ds = MMLUDataset()
        ex = Example("x", "q", "B", "mmlu", ATOMIC)
        correct, failed = ds.grade(ex, "I cannot answer that")
        assert failed and not correct

    def test_loads_real_corpus_with_atomic_label(self):
        examples = MMLUDataset().load(n=25, seed=1)
        assert len(examples) == 25
        assert all(e.expected_path == ATOMIC for e in examples)
        assert all(e.answer in "ABCD" for e in examples)

    def test_sampling_is_deterministic(self):
        a = [e.source_id for e in MMLUDataset().load(n=20, seed=7)]
        b = [e.source_id for e in MMLUDataset().load(n=20, seed=7)]
        c = [e.source_id for e in MMLUDataset().load(n=20, seed=8)]
        assert a == b and a != c


class TestTriviaQAGrading:
    def test_alias_counts_as_correct(self):
        ds = TriviaQADataset()
        ex = Example("x", "q", "David Seville", "triviaqa", LOOKUP, aliases=("Ross Bagdasarian",))
        assert ds.grade(ex, "Ross Bagdasarian")[0]

    def test_empty_response_is_extraction_failure(self):
        ds = TriviaQADataset()
        ex = Example("x", "q", "David Seville", "triviaqa", LOOKUP)
        assert ds.grade(ex, "   ") == (False, True)


class TestHotpotQAGrading:
    def test_yes_no_requires_exact_match(self):
        """Partial credit on yes/no comparison items would inflate accuracy."""
        ds = HotpotQADataset()
        ex = Example("x", "q", "yes", "hotpotqa", DECOMPOSE)
        assert ds.grade(ex, "yes")[0]
        assert not ds.grade(ex, "Actually the nationalities differ")[0]

    def test_partial_credit_for_entity_answers(self):
        ds = HotpotQADataset()
        ex = Example("x", "q", "Barack Obama", "hotpotqa", DECOMPOSE)
        assert ds.grade(ex, "Barack Hussein Obama")[0]


class TestRegistry:
    def test_every_dataset_declares_a_distinct_expected_path(self):
        """One dataset per routing destination — that is the point of the suite."""
        paths = {cls.name: cls.expected_path for cls in DATASETS.values()}
        assert paths == {"mmlu": ATOMIC, "triviaqa": LOOKUP, "hotpotqa": DECOMPOSE}
        assert len(set(paths.values())) == 3

    def test_get_dataset_rejects_unknown(self):
        with pytest.raises(ValueError, match="unknown dataset"):
            get_dataset("nope")

    def test_every_dataset_has_a_system_prompt(self):
        assert all(get_dataset(n).system_prompt() for n in DATASETS)
