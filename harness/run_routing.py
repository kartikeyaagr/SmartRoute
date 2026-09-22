#!/usr/bin/env python
"""
Run the mixed corpus through the router and measure what it actually cost.

    uv run --extra ml harness/run_routing.py --n 200 --arm routed
    uv run --extra ml harness/run_routing.py --n 200 --arm always-frontier
    CATALOG_PATH=models.anthropic.yaml uv run --extra ml harness/run_routing.py --n 200

Arms
----
    routed            the two-layer decision layer (the thing under test)
    always-cheap      floor: every prompt to the cheapest tier
    always-middle     what you would do with no router at all
    always-frontier   baseline: every prompt to the most expensive tier

Comparing `routed` against `always-frontier` gives the cost reduction; comparing it
against `always-middle` answers the more honest question — does routing beat simply
picking one good default and never thinking again?

Results append per-prompt to JSONL as they complete, so a crash mid-run loses one
result rather than the batch.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

ARMS = ["routed", "always-cheap", "always-middle", "always-frontier"]
RESULTS_DIR = Path(__file__).parent / "results"

# Pinned so a rerun is comparable. Without temperature=0 the same prompt can change
# answer between arms and the accuracy delta becomes noise. (Models that reject
# temperature have it stripped by the catalog — see ModelSpec.unsupported_params.)
#
# max_tokens must be generous enough never to bind. At 512 the stronger models hit the
# cap on 108/200 prompts while the cheapest hit it on 8, so their answers were truncated
# before the graded content appeared and they scored *worse* — an artifact of the
# budget, not a quality difference. Opus additionally returned 11 empty responses,
# having spent the whole budget on reasoning tokens before emitting any text.
GEN_PARAMS = {"temperature": 0, "max_tokens": 2000}


@dataclass
class PromptResult:
    source_id: str
    dataset: str
    expected_path: str
    category: str = ""
    route_path: str = ""
    gate_reason: str = ""
    p_lookup: float | None = None
    p_decompose: float | None = None
    final_model: str = ""
    cascade_path: list[str] = field(default_factory=list)
    subtask_count: int = 0
    is_correct: bool = False
    extraction_failed: bool = False
    response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None


_write_lock = asyncio.Lock()


async def _append(path: Path, result: PromptResult) -> None:
    async with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")


async def run_one(example, dataset, arm, router, catalog, sem, out_path) -> PromptResult:
    from smartroute.providers import ProviderError, call_model

    result = PromptResult(
        source_id=example.source_id,
        dataset=example.dataset,
        expected_path=example.expected_path,
        category=example.metadata.get("category", ""),
    )
    messages = [
        {"role": "system", "content": dataset.system_prompt()},
        {"role": "user", "content": example.prompt},
    ]

    async with sem:
        t0 = time.perf_counter()
        try:
            if arm == "routed":
                content, decision = await router.route_async(messages, request_id=example.source_id)
                result.route_path = decision.route_path
                result.gate_reason = decision.gate_reason
                result.p_lookup = decision.p_lookup
                result.p_decompose = decision.p_decompose
                result.final_model = decision.final_model
                result.cascade_path = decision.cascade_path
                result.subtask_count = decision.subtask_count
                result.input_tokens = decision.input_tokens
                result.output_tokens = decision.output_tokens
                result.cost_usd = decision.estimated_cost_usd
            else:
                role = arm.replace("always-", "")
                spec = catalog.by_role(role)
                response = await call_model(spec.id, messages, spec=spec, **GEN_PARAMS)
                content = response.content
                result.route_path = role
                result.final_model = spec.id
                result.cascade_path = [spec.id]
                result.input_tokens = response.input_tokens
                result.output_tokens = response.output_tokens
                result.cost_usd = response.estimated_cost_usd

            result.response = content[:500]
            result.is_correct, result.extraction_failed = dataset.grade(example, content)
        except (ProviderError, Exception) as exc:
            result.error = f"{type(exc).__name__}: {exc}"[:300]
        result.latency_ms = (time.perf_counter() - t0) * 1000

    await _append(out_path, result)
    return result


def summarise(arm: str, results: list[PromptResult]) -> dict:
    ok = [r for r in results if r.error is None]
    errors = len(results) - len(ok)
    evaluable = [r for r in ok if not r.extraction_failed]
    latencies = [r.latency_ms for r in ok] or [0.0]

    by_dataset = {}
    for r in ok:
        key = r.category or r.dataset
        d = by_dataset.setdefault(key, {"n": 0, "correct": 0, "cost": 0.0, "paths": {}})
        d["n"] += 1
        d["correct"] += int(r.is_correct)
        d["cost"] += r.cost_usd
        d["paths"][r.route_path] = d["paths"].get(r.route_path, 0) + 1

    return {
        "arm": arm,
        "n": len(results),
        "errors": errors,
        "accuracy": (sum(r.is_correct for r in evaluable) / len(evaluable)) if evaluable else 0.0,
        "extraction_failures": sum(r.extraction_failed for r in ok),
        # Errored prompts contribute whatever they spent before failing. Dropping them
        # would make a failure-prone arm look cheaper, which is how you accidentally
        # conclude the broken thing is the efficient one.
        "total_cost_usd": sum(r.cost_usd for r in results),
        "cost_per_prompt": sum(r.cost_usd for r in results) / max(len(results), 1),
        "p50_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * len(latencies)) - 1] if len(latencies) > 1 else latencies[0],
        "total_tokens": sum(r.input_tokens + r.output_tokens for r in results),
        "by_dataset": by_dataset,
    }


def render(summary: dict) -> None:
    print(f"\n{'=' * 78}\n  ARM: {summary['arm']}   n={summary['n']}   errors={summary['errors']}\n{'=' * 78}")
    print(f"  accuracy          {summary['accuracy']:.1%}   (extraction failures: {summary['extraction_failures']})")
    print(f"  total cost        ${summary['total_cost_usd']:.4f}   (${summary['cost_per_prompt']:.6f}/prompt)")
    print(f"  latency           p50 {summary['p50_latency_ms']:.0f}ms   p95 {summary['p95_latency_ms']:.0f}ms")
    print(f"\n  {'dataset':12s} {'n':>4s} {'accuracy':>9s} {'cost':>10s}   paths taken")
    for name, d in sorted(summary["by_dataset"].items()):
        paths = ", ".join(f"{k}={v}" for k, v in sorted(d["paths"].items(), key=lambda x: -x[1]))
        acc = d["correct"] / d["n"] if d["n"] else 0
        print(f"  {name:12s} {d['n']:4d} {acc:8.1%} ${d['cost']:9.4f}   {paths}")


def compare(routed: dict, baselines: list[dict]) -> None:
    print(f"\n{'=' * 78}\n  ROUTED vs BASELINES\n{'=' * 78}")
    print(f"  {'arm':18s} {'cost/prompt':>13s} {'accuracy':>10s} {'vs routed cost':>16s}")
    r_cost, r_acc = routed["cost_per_prompt"], routed["accuracy"]
    print(f"  {routed['arm']:18s} ${r_cost:12.6f} {r_acc:9.1%} {'—':>16s}")
    for b in baselines:
        delta = (1 - r_cost / b["cost_per_prompt"]) * 100 if b["cost_per_prompt"] else 0.0
        verdict = f"{delta:+.1f}%"
        print(f"  {b['arm']:18s} ${b['cost_per_prompt']:12.6f} {b['accuracy']:9.1%} "
              f"{'routed is ' + verdict:>16s}")
    print("\n  (negative = routed is cheaper than that baseline)")


async def run_arm(arm, examples_and_datasets, catalog, concurrency, stamp) -> dict:
    from smartroute.router import Router

    router = Router(catalog=catalog) if arm == "routed" else None
    out_path = RESULTS_DIR / f"routing_{arm}_{stamp}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        run_one(ex, ds, arm, router, catalog, sem, out_path)
        for ex, ds in examples_and_datasets
    ]
    results = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await coro)
        if i % 25 == 0 or i == len(tasks):
            spent = sum(r.cost_usd for r in results)
            print(f"    {arm}: {i}/{len(tasks)}  spent ${spent:.4f}", flush=True)
    return summarise(arm, results)


def estimate(catalog, n: int, arms: list[str]) -> float:
    """Rough upper bound so nobody starts a run without knowing the bill."""
    per_prompt = {
        "always-cheap": catalog.by_role("cheap").cost(150, 120),
        "always-middle": catalog.by_role("middle").cost(150, 120),
        "always-frontier": catalog.by_role("frontier").cost(150, 120),
        # routed: mostly middle, some cheap, ~15% decompose at ~4 calls
        "routed": 0.85 * catalog.by_role("middle").cost(150, 120)
        + 0.15 * 4 * catalog.by_role("middle").cost(150, 120),
    }
    return sum(per_prompt[a] for a in arms) * n


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="prompts total, split across the corpora in use")
    ap.add_argument("--datasets", type=str, default=None,
                    help="comma-separated corpora (default: the synthetic 200)")
    ap.add_argument("--arm", choices=ARMS + ["all"], default="routed")
    ap.add_argument("--catalog", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cost-ceiling", type=float, default=5.0)
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from harness.corpora import mixed_stream
    from smartroute.catalog import load_catalog, set_catalog

    catalog = load_catalog(args.catalog) if args.catalog else load_catalog()
    set_catalog(catalog)

    missing = catalog.missing_credentials()
    if missing:
        sys.exit(f"missing credentials: {', '.join(missing)}")

    arms = ARMS if args.arm == "all" else [args.arm]
    from harness.corpora import DEFAULT_DATASETS

    names = args.datasets.split(",") if args.datasets else DEFAULT_DATASETS
    # n is the total, so split it across however many corpora are in play.
    per_dataset = max(1, args.n // len(names))
    pairs = mixed_stream(per_dataset=per_dataset, seed=args.seed, names=names)

    print(f"catalog: {[f'{m.alias}={m.id}' for m in catalog.tiers]}")
    print(f"fingerprint: {catalog.fingerprint()}")
    print(f"corpora: {names}")
    print(f"prompts: {len(pairs)} ({per_dataset} per corpus)   arms: {arms}")

    projected = estimate(catalog, len(pairs), arms)
    print(f"estimated cost: ${projected:.2f}  (ceiling ${args.cost_ceiling:.2f})")
    if args.dry_run:
        return
    if projected > args.cost_ceiling:
        sys.exit(f"estimate ${projected:.2f} exceeds ceiling ${args.cost_ceiling:.2f} — raise --cost-ceiling to proceed")
    if not args.yes:
        if input("proceed? [y/N] ").strip().lower() != "y":
            return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summaries = []
    for arm in arms:
        print(f"\n--- running {arm} ---")
        summaries.append(await run_arm(arm, pairs, catalog, args.concurrency, stamp))

    for s in summaries:
        render(s)

    routed = next((s for s in summaries if s["arm"] == "routed"), None)
    if routed and len(summaries) > 1:
        compare(routed, [s for s in summaries if s["arm"] != "routed"])

    report = RESULTS_DIR / f"routing_summary_{stamp}.json"
    report.write_text(json.dumps(
        {"catalog_fingerprint": catalog.fingerprint(), "summaries": summaries}, indent=2
    ))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    asyncio.run(main())
