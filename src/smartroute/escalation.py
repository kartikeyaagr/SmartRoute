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

import logging
from dataclasses import dataclass

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

    async def check(self, prompt: str, response: str) -> VerifyResult:
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
