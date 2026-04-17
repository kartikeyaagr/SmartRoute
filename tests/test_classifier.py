"""
Classifier unit tests.

KeywordBaselineClassifier tests run with no dependencies.
DeBERTaNLIClassifier tests mock the transformers pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

from smartroute.classifier import (
    DeBERTaNLIClassifier,
    DifficultyTier,
    KeywordBaselineClassifier,
    _extract_prompt,
    get_classifier,
)


def msg(content: str) -> list[dict]:
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# KeywordBaselineClassifier
# ---------------------------------------------------------------------------


class TestKeywordClassifier:
    def setup_method(self):
        self.clf = KeywordBaselineClassifier()

    def test_backend_is_keyword(self):
        assert self.clf.backend() == "keyword"

    def test_simple_factual_easy(self):
        tier, conf = self.clf.classify(msg("What is the capital of France?"))
        assert tier == DifficultyTier.EASY
        assert conf > 0.5

    def test_who_question_easy(self):
        tier, conf = self.clf.classify(msg("Who was the first US president?"))
        assert tier == DifficultyTier.EASY

    def test_short_prompt_easy(self):
        tier, _ = self.clf.classify(msg("Define photosynthesis."))
        assert tier == DifficultyTier.EASY

    def test_explain_medium(self):
        tier, conf = self.clf.classify(msg("Explain how photosynthesis works."))
        assert tier == DifficultyTier.MEDIUM
        assert conf > 0.5

    def test_calculate_medium(self):
        tier, _ = self.clf.classify(msg("Calculate the area of a circle with radius 5."))
        assert tier == DifficultyTier.MEDIUM

    def test_prove_hard(self):
        tier, conf = self.clf.classify(msg("Prove that the square root of 2 is irrational."))
        assert tier == DifficultyTier.HARD
        assert conf > 0.7

    def test_formal_logic_hard(self):
        tier, _ = self.clf.classify(msg("Using predicate calculus, formalize the sentence."))
        assert tier == DifficultyTier.HARD

    def test_math_symbols_hard(self):
        tier, _ = self.clf.classify(msg("Evaluate ∫(x²+1)dx from 0 to 5."))
        assert tier == DifficultyTier.HARD

    def test_long_explain_hard(self):
        long_prompt = "Explain " + " ".join(["the complex relationship between"] * 30) + " evaluate all implications"
        tier, _ = self.clf.classify(msg(long_prompt))
        assert tier == DifficultyTier.HARD

    def test_empty_messages_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.clf.classify([])

    def test_no_user_message_raises(self):
        with pytest.raises(ValueError):
            self.clf.classify([{"role": "assistant", "content": "Hi"}])

    def test_returns_medium_by_default(self):
        # 17-word prompt with no keyword matches → falls to default MEDIUM branch
        tier, conf = self.clf.classify(msg(
            "Tell me something interesting about the ocean and how ocean currents influence global weather and climate."
        ))
        assert tier == DifficultyTier.MEDIUM


# ---------------------------------------------------------------------------
# _extract_prompt
# ---------------------------------------------------------------------------


def test_extract_prompt_last_user_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    assert _extract_prompt(messages) == "Second question"


def test_extract_prompt_skips_empty_user():
    messages = [
        {"role": "user", "content": ""},
        {"role": "user", "content": "Real question"},
    ]
    assert _extract_prompt(messages) == "Real question"


# ---------------------------------------------------------------------------
# DeBERTaNLIClassifier (mocked pipeline)
# ---------------------------------------------------------------------------


def _make_mock_pipeline(label: str, score: float = 0.82):
    """Build a mock transformers pipeline that returns a fixed label + score."""
    mock_pipe = MagicMock()
    # Labels are returned sorted by score descending
    all_labels = [
        "is a simple factual question",
        "requires moderate reasoning",
        "requires complex expert reasoning",
    ]
    # Put the winning label first
    other_labels = [l for l in all_labels if l != label]
    mock_pipe.return_value = {
        "labels": [label] + other_labels,
        "scores": [score, (1 - score) / 2, (1 - score) / 2],
    }
    return mock_pipe


def _deberta_clf_with_mock_pipeline(label: str, score: float = 0.82) -> DeBERTaNLIClassifier:
    """Construct a DeBERTaNLIClassifier with a mocked pipeline (skips download + warmup)."""
    clf = DeBERTaNLIClassifier.__new__(DeBERTaNLIClassifier)
    clf._fallback = KeywordBaselineClassifier()
    clf._using_fallback = False
    clf._pipeline = _make_mock_pipeline(label, score)
    return clf


class TestDeBERTaClassifier:
    def test_easy_label_maps_to_easy_tier(self):
        clf = _deberta_clf_with_mock_pipeline("is a simple factual question", 0.90)
        tier, conf = clf.classify(msg("What is 2+2?"))
        assert tier == DifficultyTier.EASY
        assert conf == pytest.approx(0.90)

    def test_medium_label_maps_to_medium_tier(self):
        clf = _deberta_clf_with_mock_pipeline("requires moderate reasoning", 0.75)
        tier, conf = clf.classify(msg("Explain how neural networks learn."))
        assert tier == DifficultyTier.MEDIUM
        assert conf == pytest.approx(0.75)

    def test_hard_label_maps_to_hard_tier(self):
        clf = _deberta_clf_with_mock_pipeline("requires complex expert reasoning", 0.88)
        tier, conf = clf.classify(msg("Prove Fermat's Last Theorem."))
        assert tier == DifficultyTier.HARD
        assert conf == pytest.approx(0.88)

    def test_fallback_used_when_pipeline_is_none(self):
        clf = DeBERTaNLIClassifier.__new__(DeBERTaNLIClassifier)
        clf._fallback = KeywordBaselineClassifier()
        clf._using_fallback = True
        clf._pipeline = None
        tier, _ = clf.classify(msg("What is the capital of France?"))
        assert tier == DifficultyTier.EASY

    def test_backend_deberta_when_pipeline_loaded(self):
        clf = _deberta_clf_with_mock_pipeline("is a simple factual question")
        assert clf.backend() == "deberta"

    def test_backend_keyword_when_fallback(self):
        clf = DeBERTaNLIClassifier.__new__(DeBERTaNLIClassifier)
        clf._fallback = KeywordBaselineClassifier()
        clf._using_fallback = True
        clf._pipeline = None
        assert clf.backend() == "keyword"

    def test_warmup_threshold_triggers_fallback(self):
        """If warmup median > 2000ms, DeBERTa falls back to keyword."""
        slow_pipe = MagicMock()
        slow_pipe.return_value = {
            "labels": ["is a simple factual question", "requires moderate reasoning", "requires complex expert reasoning"],
            "scores": [0.9, 0.05, 0.05],
        }

        # perf_counter called once before and once after each warmup call (3 calls = 6 perf_counter calls)
        # Make each call appear to take 1500ms → median of calls 2-3 = 1500ms > 2000ms threshold? No...
        # Actually 1500ms < 2000ms. Make them 2500ms each.
        time_values = []
        t = 0.0
        for _ in range(3):
            time_values.append(t)       # before call
            t += 2.5                    # 2500ms per warmup call
            time_values.append(t)       # after call

        with patch("smartroute.classifier.hf_pipeline", return_value=slow_pipe):
            with patch("smartroute.classifier.time.perf_counter", side_effect=time_values):
                clf = DeBERTaNLIClassifier()

        assert clf._using_fallback is True
        assert clf.backend() == "keyword"

    def test_prompt_truncated_to_2048_chars(self):
        """Long prompts are truncated before being passed to the pipeline."""
        clf = _deberta_clf_with_mock_pipeline("requires moderate reasoning")
        long_prompt = "x" * 5000
        clf.classify(msg(long_prompt))
        call_args = clf._pipeline.call_args[0]
        assert len(call_args[0]) == 2048


# ---------------------------------------------------------------------------
# get_classifier factory
# ---------------------------------------------------------------------------


def test_get_classifier_keyword():
    clf = get_classifier("keyword")
    assert isinstance(clf, KeywordBaselineClassifier)


def test_get_classifier_default_is_keyword():
    clf = get_classifier()
    assert isinstance(clf, KeywordBaselineClassifier)


def test_get_classifier_deberta_returns_deberta_type():
    """Factory returns DeBERTaNLIClassifier for 'deberta' (init may fall back internally)."""
    fast_pipe = _make_mock_pipeline("is a simple factual question")
    # 3 warmup calls, each ~10ms: perf_counter alternates start/end for 3 iterations
    time_values = []
    t = 0.0
    for _ in range(3):
        time_values.append(t)
        t += 0.01  # 10ms per call — well under 2000ms threshold
        time_values.append(t)

    with patch("smartroute.classifier.hf_pipeline", return_value=fast_pipe):
        with patch("smartroute.classifier.time.perf_counter", side_effect=time_values):
            clf = get_classifier("deberta")
    assert isinstance(clf, DeBERTaNLIClassifier)
