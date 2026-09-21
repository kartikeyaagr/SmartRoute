"""
Layer 2: split a request into sub-tasks, route each to the cheapest tier that can do
it, then synthesise.

This layer is only reached when the gate in decision.py has already decided that
splitting is worth paying for. It is expensive: decompose + synthesise alone costs
about 2.4x simply answering with the middle model, before a single sub-task runs. Its
whole justification is doing frontier-grade work without a frontier call, so it is
wrapped in guards that give up the moment that stops being true.

    plan = middle.decompose(prompt)         one call
      |-- malformed?  -> answer directly with middle
      |-- atomic?     -> answer directly with middle   (the model's own veto)
      |-- too costly? -> one frontier call             (projected-cost bail-out)
      |
      +-- dispatch sub-tasks concurrently across tiers
          +-- middle.synthesise(results)    one call

Sub-tasks never decompose again. Depth is fixed at 1: the cost of a plan is linear in
its fan-out, and recursion turns a bounded overhead into an unbounded one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from smartroute.catalog import Catalog, ModelSpec, get_catalog
from smartroute.config import settings
from smartroute.providers import ProviderError, call_model

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 6
VALID_TIERS = ("cheap", "middle", "frontier")

_DECOMPOSE_SYSTEM = """\
You break a request into the smallest set of independent sub-tasks that can be \
answered separately and then combined.

Reply with ONLY a JSON object, no prose:
{"atomic": false, "subtasks": [{"task": "...", "tier": "cheap"}]}

Rules:
- If the request can be answered well in one step, reply {"atomic": true, "subtasks": []}.
- Never emit more than %d sub-tasks.
- Each sub-task must be self-contained: it is answered without seeing the others.
- "tier" is how much model the sub-task needs:
    "cheap"    simple factual lookup or extraction
    "middle"   ordinary reasoning, summarising, or writing
    "frontier" genuinely hard reasoning that smaller models would get wrong
- Prefer "cheap". Choosing "frontier" for most sub-tasks means this request should \
not have been split at all.""" % MAX_SUBTASKS

_SYNTHESIS_SYSTEM = """\
You are given a user's request and the answers to its sub-tasks. Write the final \
answer to the original request using those results. Do not mention the sub-tasks or \
that the work was split."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class SubTask:
    task: str
    tier: str

    @property
    def is_valid(self) -> bool:
        return bool(self.task.strip()) and self.tier in VALID_TIERS


@dataclass
class Plan:
    atomic: bool
    subtasks: list[SubTask] = field(default_factory=list)
    malformed: bool = False

    @property
    def should_execute(self) -> bool:
        """Splitting into one sub-task is just a more expensive direct answer."""
        return not self.atomic and not self.malformed and len(self.subtasks) > 1


@dataclass
class DecompositionResult:
    content: str
    route: str                      # "decomposed" | "direct-middle" | "bailout-frontier"
    reason: str
    subtasks: list[SubTask] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def parse_plan(text: str) -> Plan:
    """
    Parse the planner's JSON, tolerantly.

    A malformed plan is not an error worth failing a user request over — it degrades
    to answering directly with the middle model, which is what would have happened
    had the gate not fired.
    """
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return Plan(atomic=False, malformed=True)
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Plan(atomic=False, malformed=True)

    if raw.get("atomic"):
        return Plan(atomic=True)

    subtasks = []
    for item in raw.get("subtasks") or []:
        if not isinstance(item, dict):
            continue
        candidate = SubTask(
            task=str(item.get("task", "")),
            tier=str(item.get("tier", "middle")).lower(),
        )
        if candidate.is_valid:
            subtasks.append(candidate)

    if not subtasks:
        return Plan(atomic=False, malformed=True)
    return Plan(atomic=False, subtasks=subtasks[:MAX_SUBTASKS])


class Decomposer:
    def __init__(
        self,
        catalog: Catalog | None = None,
        concurrency: int = 4,
        max_subtasks: int = MAX_SUBTASKS,
    ) -> None:
        self._catalog = catalog or get_catalog()
        self._concurrency = concurrency
        self._max_subtasks = max_subtasks

    # -- costing -----------------------------------------------------------

    def price_plan(self, prompt: str, plan: Plan) -> float:
        """
        What executing this plan will cost, using the tiers the planner asked for.

        Estimated from the plan rather than measured, because the point is to decide
        whether to run it at all.
        """
        approx_in = len(prompt) // 4 + 60
        approx_out = 200
        subtask_cost = sum(
            self._catalog.by_role(s.tier).cost(approx_in, approx_out) for s in plan.subtasks
        )
        middle = self._catalog.by_role("middle")
        synthesis = middle.cost(len(prompt) // 4 + len(plan.subtasks) * approx_out, 400)
        return subtask_cost + synthesis

    def _frontier_cost(self, prompt: str) -> float:
        return self._catalog.by_role("frontier").cost(len(prompt) // 4, 300)

    # -- execution ---------------------------------------------------------

    async def run(self, prompt: str, messages: list[dict]) -> DecompositionResult:
        result = DecompositionResult(content="", route="decomposed", reason="")

        plan_text, plan_cost = await self._call("middle", [
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user", "content": prompt},
        ], result)
        plan = parse_plan(plan_text)

        if plan.malformed:
            logger.warning("planner returned an unparseable plan — answering directly")
            return await self._direct(prompt, messages, result, "middle", "plan was malformed")

        if not plan.should_execute:
            reason = "planner judged the request atomic" if plan.atomic else "plan had a single sub-task"
            return await self._direct(prompt, messages, result, "middle", reason)

        plan.subtasks = plan.subtasks[: self._max_subtasks]

        # Projected-cost bail-out. Decomposition exists to avoid a frontier call; if
        # the plan the model produced costs more than that call, the premise is gone.
        projected = self.price_plan(prompt, plan)
        frontier = self._frontier_cost(prompt)
        if projected >= frontier:
            logger.info(
                "plan projected at $%.6f >= one frontier call $%.6f — bailing out",
                projected, frontier,
            )
            return await self._direct(
                prompt, messages, result, "frontier",
                f"plan projected ${projected:.6f} >= frontier ${frontier:.6f}",
            )

        answers = await self._dispatch(plan, result)

        digest = "\n\n".join(
            f"SUB-TASK: {s.task}\nRESULT: {a}" for s, a in zip(plan.subtasks, answers)
        )
        content, _ = await self._call("middle", [
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": f"ORIGINAL REQUEST:\n{prompt}\n\n{digest}"},
        ], result)

        result.content = content
        result.subtasks = plan.subtasks
        result.reason = f"{len(plan.subtasks)} sub-tasks, projected ${projected:.6f}"
        return result

    async def _dispatch(self, plan: Plan, result: DecompositionResult) -> list[str]:
        """Sub-tasks are independent by construction, so they run concurrently."""
        semaphore = asyncio.Semaphore(self._concurrency)

        async def one(subtask: SubTask) -> str:
            async with semaphore:
                try:
                    text, _ = await self._call(
                        subtask.tier, [{"role": "user", "content": subtask.task}], result
                    )
                    return text
                except ProviderError as exc:
                    # One failed sub-task must not sink the whole request; synthesis
                    # is told what is missing and works with the rest.
                    logger.warning("sub-task failed on %s: %s", subtask.tier, exc)
                    return f"[unavailable: {exc}]"

        return await asyncio.gather(*(one(s) for s in plan.subtasks))

    async def _direct(
        self,
        prompt: str,
        messages: list[dict],
        result: DecompositionResult,
        tier: str,
        reason: str,
    ) -> DecompositionResult:
        content, _ = await self._call(tier, messages, result)
        result.content = content
        result.route = "direct-middle" if tier == "middle" else "bailout-frontier"
        result.reason = reason
        return result

    async def _call(
        self, tier: str, messages: list[dict], result: DecompositionResult
    ) -> tuple[str, float]:
        spec: ModelSpec = self._catalog.by_role(tier)
        response = await call_model(
            spec.id, messages, timeout_s=settings.model_timeout_s, spec=spec
        )
        result.models_used.append(spec.id)
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.cost_usd += response.estimated_cost_usd
        return response.content, response.estimated_cost_usd
