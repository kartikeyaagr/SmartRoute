"""
Escalation signal: a decision model checks the middle tier's answer.

RouterBench showed where the money is. The oracle — escalate exactly the prompts the
middle model gets wrong — beats always-frontier on accuracy at ~1/9 of its cost, while
the two-layer router, which never escalates, scores the same as always-middle. The gap
is entirely "notice the middle answer is wrong, then pay for the frontier".

So after every middle answer, one Decisions API call (Jev by default) asks a single
Noul question — does this response correctly and fully answer the request? — and gets
back P(yes). Output tokens are free and the question is typed, so there is no
parse-the-judge's-prose step and no judge that grades its own homework.

The rule
--------
Escalating costs one frontier call. Not escalating risks a wrong answer, which the
operator prices at lambda (models.yaml `lambda_wrong_answer_usd`). So escalate when

    (1 - p_correct) * lambda  >  frontier_cost

i.e. when p_correct < 1 - frontier_cost / lambda. On the shipped catalog
(lambda=$0.001, frontier ~$0.0002 per call) that is p_correct < ~0.78.

This treats the frontier answer as correct, which overstates the gain; and Jev's
probabilities are not calibrated against your traffic. Both are reasons to pin the
threshold from a labelled sample (harness/routerbench.py --jev-verify) rather than
trust the derived one.

Failure policy
--------------
If the check fails (network, quota, no key) the middle answer is returned unescalated.
The checker is an optimisation; its outage must not turn into an answer outage or a
silent jump to frontier pricing.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from smartroute.taskroute import (
    DEFAULT_DECISION_MODEL,
    DecisionsClient,
    TaskRouteConfig,
    TaskRouteError,
)

logger = logging.getLogger(__name__)

_PROMPT_CHARS = 12_000
_RESPONSE_CHARS = 12_000

QUESTION = {
    "answers_correctly": {
        "type": "noul",
        "instructions": (
            "Does the assistant's response correctly and completely answer the user's message?"
        ),
        "criteria": {
            "true": "The response is correct, addresses every part of the request, and "
                    "contains no factual, logical or code errors.",
            "false": "The response is wrong, incomplete, evasive, off-topic, or contains an "
                     "error in facts, reasoning, arithmetic or code.",
        },
    }
}


@dataclass(frozen=True)
class VerifyResult:
    p_correct: float | None   # None when the check itself failed
    escalate: bool
    cost_usd: float
    reason: str


def derived_threshold(frontier_cost_usd: float, lambda_wrong_usd: float) -> float:
    """Lowest p_correct at which keeping the middle answer beats paying for frontier."""
    if lambda_wrong_usd <= 0:
        return 0.0  # wrong answers cost nothing: never escalate
    return max(0.0, 1.0 - frontier_cost_usd / lambda_wrong_usd)


class JevVerifier:
    def __init__(
        self,
        api_key: str,
        threshold: float,
        model: str = DEFAULT_DECISION_MODEL,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        if not api_key:
            raise TaskRouteError("OPENROUTER_API_KEY is required for Jev escalation")
        self.threshold = threshold
        self._decisions = DecisionsClient(
            api_key, TaskRouteConfig(decision_model=model, timeout_s=timeout_s), client=client
        )

    async def check(self, prompt: str, response: str, messages: list[dict] | None = None) -> VerifyResult:
        if not response.strip():
            return VerifyResult(p_correct=0.0, escalate=True, cost_usd=0.0, reason="empty middle answer")
        try:
            answers, cost = await self._decisions.decide(
                state={
                    "user_message": prompt[:_PROMPT_CHARS],
                    "assistant_response": response[:_RESPONSE_CHARS],
                },
                questions=QUESTION,
            )
            p = float(answers["answers_correctly"]["noul"])
        except (TaskRouteError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Jev verification failed: %s — keeping middle answer", exc)
            return VerifyResult(p_correct=None, escalate=False, cost_usd=0.0,
                                reason=f"verify failed ({exc}); kept middle")
        escalate = p < self.threshold
        return VerifyResult(
            p_correct=p,
            escalate=escalate,
            cost_usd=cost,
            reason=f"jev p_correct={p:.2f} {'<' if escalate else '>='} {self.threshold:.2f}",
        )


# ---------------------------------------------------------------------------
# Learned checker: P(correct) from embeddings of the prompt and the answer
# ---------------------------------------------------------------------------

DEFAULT_CHECKER_PATH = Path(__file__).parent / "data" / "checker.joblib"


def pair_features(prompt_emb, response_emb):
    """[p, r, p*r, |p-r|] — the standard pair encoding for a pair classifier."""
    import numpy as np

    return np.hstack([prompt_emb, response_emb, prompt_emb * response_emb, np.abs(prompt_emb - response_emb)])


def tail(text: str, chars: int) -> str:
    """The question sits at the end of few-shot prompts; a 256-token encoder sees the start."""
    return text[-chars:] if chars and len(text) > chars else text


class LearnedVerifier:
    """
    Classifier over frozen MiniLM embeddings of (prompt, answer), trained on graded
    answers by harness/train_checker.py.

    It is only as good as the match between its training answers and the model it
    checks: a checker fitted on Mixtral's RouterBench answers has not seen gpt-oss.
    Train it on labelled answers from the middle model you actually run.
    """

    def __init__(self, threshold: float, path: Path | None = None) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise TaskRouteError("learned checker needs the ml extra: uv sync --extra ml") from exc

        path = path or DEFAULT_CHECKER_PATH
        if not path.exists():
            raise TaskRouteError(f"no trained checker at {path}; run harness/train_checker.py")
        bundle = joblib.load(path)
        self._head = bundle["head"]
        self._encoder_name = bundle["encoder"]
        self._tail_chars = int(bundle.get("tail_chars", 1000))
        self.trained_on = bundle.get("trained_on", "")
        self.threshold = threshold
        self._encoder = None

    def _p_correct(self, prompt: str, response: str) -> float:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._encoder_name)
        e = self._encoder.encode(
            [tail(prompt, self._tail_chars), tail(response, self._tail_chars)],
            normalize_embeddings=True,
        )
        return float(self._head.predict_proba(pair_features(e[:1], e[1:2]))[0][1])

    async def check(self, prompt: str, response: str, messages: list[dict] | None = None) -> VerifyResult:
        if not response.strip():
            return VerifyResult(p_correct=0.0, escalate=True, cost_usd=0.0, reason="empty middle answer")
        try:
            p = await asyncio.to_thread(self._p_correct, prompt, response)
        except Exception as exc:  # inference failure must not fail the request
            logger.warning("learned checker failed: %s — keeping middle answer", exc)
            return VerifyResult(None, False, 0.0, f"learned check failed ({exc}); kept middle")
        escalate = p < self.threshold
        return VerifyResult(p, escalate, 0.0,
                            f"learned p_correct={p:.2f} {'<' if escalate else '>='} {self.threshold:.2f}")


class EnsembleVerifier:
    """Mean P(correct) of the checkers that answered; escalates on the mean."""

    def __init__(self, verifiers: list, threshold: float) -> None:
        self._verifiers = verifiers
        self.threshold = threshold

    async def check(self, prompt: str, response: str, messages: list[dict] | None = None) -> VerifyResult:
        results = await asyncio.gather(*(v.check(prompt, response, messages) for v in self._verifiers))
        cost = sum(r.cost_usd for r in results)
        ps = [r.p_correct for r in results if r.p_correct is not None]
        if not ps:
            return VerifyResult(None, False, cost, "all checkers failed; kept middle")
        p = sum(ps) / len(ps)
        escalate = p < self.threshold
        parts = ", ".join(r.reason for r in results)
        return VerifyResult(p, escalate, cost,
                            f"ensemble p_correct={p:.2f} {'<' if escalate else '>='} {self.threshold:.2f} [{parts}]")


# ---------------------------------------------------------------------------
# Self-consistency: re-ask, and escalate when the answers disagree
# ---------------------------------------------------------------------------
#
# A model that knows the answer tends to give it again; one that is guessing scatters.
# So sample k more answers and use the share that agree with the original as
# P(correct). Unlike the learned checker it needs no training data, and unlike Jev it
# does not have to solve the question itself — it only has to compare answers.
#
# It costs k extra calls. Two sources:
#   middle  true self-consistency: k samples of the same model at temperature > 0
#   cheap   cross-model agreement: k samples of the cheap tier — cheaper, and the only
#           variant RouterBench can replay (it records one answer per model)

_LETTER = re.compile(
    r"(?i:answer(?: is)?|option|choice)\s*[:\-]?\s*\(?([A-Ea-e])\)?(?![A-Za-z])"
    r"|\[([A-E])\]"
    # a bare letter on its own line, or "B." / "(B)" / "B)" — but not "A man ..."
    r"|^\s*\(?([A-E])(?:\)|\.|:)?\s*$"
    r"|^\s*\(?([A-E])[).:]",
    re.MULTILINE,
)
_GSM_FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_WORD = re.compile(r"[a-z0-9]+")


def extract_answer(text: str) -> tuple[str, str]:
    """(kind, value): ("letter", "B") | ("number", "42") | ("text", "normalised words")."""
    t = (text or "").strip()
    if "```" in t or re.search(r"^\s*(def|class|import|function)\s", t, re.MULTILINE):
        return "text", " ".join(_WORD.findall(t.lower()))  # code: compare as text, not by its last number
    m = _LETTER.search(t[:400])
    if m:
        return "letter", next(g for g in m.groups() if g).upper()
    m = _GSM_FINAL.search(t)
    nums = [m.group(1)] if m else _NUMBER.findall(t)
    if nums and len(t) < 2000:
        n = nums[-1].replace(",", "").rstrip(".")
        try:
            return "number", f"{float(n):g}"
        except ValueError:
            pass
    return "text", " ".join(_WORD.findall(t.lower()))


def _token_f1(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return float(ta == tb)
    common = sum(min(ta.count(w), tb.count(w)) for w in set(ta))
    if common == 0:
        return 0.0
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def answers_agree(a: str, b: str, text_f1: float = 0.6) -> bool:
    """Same extracted letter/number, or free text with token F1 >= text_f1."""
    ka, va = extract_answer(a)
    kb, vb = extract_answer(b)
    if ka != "text" and kb != "text":
        return ka == kb and va == vb
    if ka != "text" or kb != "text":
        return False  # one gave a definite letter/number, the other did not
    return _token_f1(va, vb) >= text_f1


# (messages, n) -> list of (answer text, cost in USD)
Sampler = Callable[[list[dict], int], Awaitable[list[tuple[str, float]]]]


class SelfConsistencyVerifier:
    def __init__(self, sampler: Sampler, threshold: float, samples: int = 2, source: str = "middle") -> None:
        if samples < 1:
            raise ValueError("self-consistency needs at least one extra sample")
        self._sample = sampler
        self.threshold = threshold
        self.samples = samples
        self.source = source

    async def check(self, prompt: str, response: str, messages: list[dict] | None = None) -> VerifyResult:
        if not response.strip():
            return VerifyResult(0.0, True, 0.0, "empty middle answer")
        msgs = messages or [{"role": "user", "content": prompt}]
        try:
            drawn = await self._sample(msgs, self.samples)
        except Exception as exc:
            logger.warning("self-consistency sampling failed: %s — keeping middle answer", exc)
            return VerifyResult(None, False, 0.0, f"self-consistency failed ({exc}); kept middle")
        cost = sum(c for _, c in drawn)
        answers = [a for a, _ in drawn if a.strip()]
        if not answers:
            return VerifyResult(None, False, cost, "self-consistency got no samples; kept middle")
        p = sum(answers_agree(response, a) for a in answers) / len(answers)
        escalate = p < self.threshold
        return VerifyResult(p, escalate, cost,
                            f"consistency[{self.source} x{len(answers)}] agree={p:.2f} "
                            f"{'<' if escalate else '>='} {self.threshold:.2f}")
