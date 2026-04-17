"""
Difficulty classifiers for LLM routing.

KeywordBaselineClassifier: heuristic, <1ms, no model download.
DeBERTaNLIClassifier: zero-shot NLI, ~200-800ms CPU, auto-fallbacks to keyword if too slow.
"""

import logging
import re
import statistics
import time
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)

# Module-level import so tests can patch `smartroute.classifier.hf_pipeline`
try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    hf_pipeline = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False


class DifficultyTier(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class DifficultyClassifier(ABC):
    @abstractmethod
    def classify(self, messages: list[dict]) -> tuple[DifficultyTier, float]:
        """
        Classify the difficulty of the last user message.

        Returns:
            (tier, confidence) where confidence is 0.0-1.0
        """

    @abstractmethod
    def backend(self) -> str:
        """Return 'deberta' or 'keyword'."""


def _extract_prompt(messages: list[dict]) -> str:
    """Extract the last user message content from a messages list."""
    if not messages:
        raise ValueError("messages list is empty")
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return content
    raise ValueError("No non-empty user message found in messages")


# ---------------------------------------------------------------------------
# Keyword Baseline Classifier
# ---------------------------------------------------------------------------

_HARD_KEYWORDS = re.compile(
    r"\b("
    r"prove|proof|derive|derivation|theorem|lemma|corollary|conjecture"
    r"|critique|synthesize|synthesis|evaluate the validity"
    r"|formal logic|predicate calculus|propositional logic"
    r"|differential equation|eigenvalue|eigenvector"
    r"|stochastic|bayesian inference|posterior distribution"
    r"|constitutional law|statutory interpretation|legal brief"
    r"|clinical trial|pharmacokinetics|pathophysiology"
    r")\b",
    re.IGNORECASE,
)

_MEDIUM_KEYWORDS = re.compile(
    r"\b("
    r"explain|analyze|compare|contrast|evaluate|discuss|describe the relationship"
    r"|what are the implications|why does|how does|what causes"
    r"|summarize|outline the steps|walk me through"
    r"|calculate|solve|find the value|compute|determine"
    r"|essay|argument|essay on"
    r")\b",
    re.IGNORECASE,
)

_EASY_PATTERNS = re.compile(
    r"^("
    r"what is|who is|who was|when did|when was|where is|where was"
    r"|what year|what country|what city|what language|what color"
    r"|define |what does .{1,30} mean|what does .{1,30} stand for"
    r"|how many|how much|is it true that|true or false"
    r")",
    re.IGNORECASE,
)

_MATH_SYMBOLS = re.compile(r"[∫∑∏√∂∇∞±≤≥≠≈∈∉∪∩⊂⊃∀∃]|d/dx|\bintegral\b|\blimit\b|\bderivative\b")


class KeywordBaselineClassifier(DifficultyClassifier):
    """
    Fast heuristic classifier. <1ms. No model download required.

    Heuristics (applied in order, first match wins):
    1. Hard keywords → HARD (0.85 confidence)
    2. Advanced math symbols → HARD (0.80)
    3. Long prompt (>120 words) + medium keywords → HARD (0.75)
    4. Medium keywords present → MEDIUM (0.70)
    5. Simple question patterns → EASY (0.80)
    6. Short prompt (<15 words) → EASY (0.65)
    7. Default → MEDIUM (0.55)
    """

    def classify(self, messages: list[dict]) -> tuple[DifficultyTier, float]:
        prompt = _extract_prompt(messages)
        words = prompt.split()
        word_count = len(words)

        if _HARD_KEYWORDS.search(prompt):
            return DifficultyTier.HARD, 0.85

        if _MATH_SYMBOLS.search(prompt):
            return DifficultyTier.HARD, 0.80

        if word_count > 120 and _MEDIUM_KEYWORDS.search(prompt):
            return DifficultyTier.HARD, 0.75

        if _MEDIUM_KEYWORDS.search(prompt):
            return DifficultyTier.MEDIUM, 0.70

        if _EASY_PATTERNS.match(prompt.strip()):
            return DifficultyTier.EASY, 0.80

        if word_count < 15:
            return DifficultyTier.EASY, 0.65

        return DifficultyTier.MEDIUM, 0.55

    def backend(self) -> str:
        return "keyword"


# ---------------------------------------------------------------------------
# DeBERTa NLI Classifier
# ---------------------------------------------------------------------------

_NLI_LABELS = [
    "is a simple factual question",
    "requires moderate reasoning",
    "requires complex expert reasoning",
]

_LABEL_TO_TIER = {
    "is a simple factual question": DifficultyTier.EASY,
    "requires moderate reasoning": DifficultyTier.MEDIUM,
    "requires complex expert reasoning": DifficultyTier.HARD,
}

_DEBERTA_MODEL = "cross-encoder/nli-deberta-v3-small"
_WARMUP_THRESHOLD_MS = 2000.0  # fall back to keyword if median warmup > this


class DeBERTaNLIClassifier(DifficultyClassifier):
    """
    Zero-shot NLI classifier using cross-encoder/nli-deberta-v3-small.

    Downloads ~400MB on first use. Warmup runs 3 calls at init; if median
    of calls 2-3 exceeds 2000ms, silently falls back to KeywordBaselineClassifier.

    difficulty_score = softmax probability of the predicted class.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._fallback = KeywordBaselineClassifier()
        self._using_fallback = False
        self._init_pipeline()

    def _init_pipeline(self) -> None:
        if not _TRANSFORMERS_AVAILABLE or hf_pipeline is None:
            logger.warning(
                "transformers not installed — falling back to KeywordBaselineClassifier. "
                "Install with: uv add 'smartroute[ml]'"
            )
            self._using_fallback = True
            return

        logger.info("Downloading DeBERTa model (~400MB)... (one-time, cached after first run)")
        try:
            pipe = hf_pipeline(
                "zero-shot-classification",
                model=_DEBERTA_MODEL,
                hypothesis_template="This text {}",
            )
        except Exception as e:
            logger.warning("Failed to load DeBERTa model: %s — falling back to keyword", e)
            self._using_fallback = True
            return

        # Warmup: 3 calls, measure median of calls 2 and 3 (skip cold JIT on call 1)
        warmup_text = "What is the capital of France?"
        warmup_times: list[float] = []
        for i in range(3):
            t0 = time.perf_counter()
            pipe(warmup_text, _NLI_LABELS)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if i > 0:
                warmup_times.append(elapsed_ms)

        median_ms = statistics.median(warmup_times)
        logger.info("DeBERTa warmup median (calls 2-3): %.0fms", median_ms)

        if median_ms > _WARMUP_THRESHOLD_MS:
            logger.warning(
                "DeBERTa warmup %.0fms > %.0fms threshold — falling back to KeywordBaselineClassifier. "
                "Set CLASSIFIER=deberta to force DeBERTa (may be slow).",
                median_ms,
                _WARMUP_THRESHOLD_MS,
            )
            self._using_fallback = True
            return

        self._pipeline = pipe
        logger.info("DeBERTa classifier ready (%.0fms median warmup)", median_ms)

    def classify(self, messages: list[dict]) -> tuple[DifficultyTier, float]:
        if self._using_fallback or self._pipeline is None:
            return self._fallback.classify(messages)

        prompt = _extract_prompt(messages)
        # Truncate to ~512 tokens worth of characters (rough: 4 chars/token)
        prompt_truncated = prompt[:2048]

        result = self._pipeline(prompt_truncated, _NLI_LABELS)
        # result["labels"] and result["scores"] are sorted by score descending
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        tier = _LABEL_TO_TIER[top_label]
        return tier, top_score

    def backend(self) -> str:
        return "keyword" if self._using_fallback else "deberta"


def get_classifier(classifier_type: str = "keyword") -> DifficultyClassifier:
    """
    Factory: returns the configured classifier.

    Args:
        classifier_type: "keyword" or "deberta"
    """
    if classifier_type == "deberta":
        return DeBERTaNLIClassifier()
    return KeywordBaselineClassifier()
