"""
SmartRoute MMLU Benchmark Runner — All 4 ablation modes.

Modes:
  1 / routing-only    — classifier decides tier, no verifier, cheap or frontier directly
  2 / cascade-only    — all prompts treated as MEDIUM (cheap → verifier → maybe frontier)
  3 / full            — classifier + cascade (full SmartRoute)
  4 / always-frontier — all prompts → groq/llama-3.3-70b-versatile (baseline)

Usage:
  uv run bench/run_benchmark.py --mode routing-only --dry-run
  uv run bench/run_benchmark.py --mode always-frontier --prompts 50
  uv run bench/run_benchmark.py --mode full
  uv run bench/run_benchmark.py --all-modes --prompts 50
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root so imports work when run as a script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from smartroute.classifier import DifficultyClassifier, DifficultyTier, get_classifier
from smartroute.config import settings
from smartroute.providers import ProviderError, call_model
from smartroute.router import Router, RoutingDecision

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MMLU_SYSTEM_PROMPT = "Respond with ONLY the letter A, B, C, or D on the first line."
VALID_ANSWERS = {"A", "B", "C", "D"}
DATA_PATH = Path(__file__).parent / "data" / "mmlu_500.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

_COST_PER_1K: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":              (0.00025, 0.00125),
    "gemini/gemini-2.5-flash":       (0.00015, 0.0006),
    "ollama/llama3.1":               (0.0, 0.0),
    "gpt-4o":                        (0.0025, 0.01),
    "claude-opus-4-6":               (0.015, 0.075),
    "groq/llama-3.1-8b-instant":     (0.00005, 0.00008),
    "groq/llama-3.3-70b-versatile":  (0.00059, 0.00079),
}
_AVG_MMLU_INPUT_TOKENS = 200
_AVG_OUTPUT_TOKENS = 3

ALL_MODES = ["routing-only", "cascade-only", "full", "always-frontier"]

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    source_id: str
    subject: str
    difficulty_tier: str
    correct_answer: str
    # Final response
    model_response: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    extraction_failed: bool = False
    # Cheap model response (populated when verifier ran)
    cheap_answer: str = ""
    cheap_correct: bool | None = None
    # Routing metadata
    request_id: str = ""
    final_model: str = ""
    cascade_path: list[str] = field(default_factory=list)
    verifier_score: int | None = None
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    escalated: bool = False
    classifier_backend: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Fixed-tier classifier for mode 2 (cascade-only)
# ---------------------------------------------------------------------------

class _AlwaysMediumClassifier(DifficultyClassifier):
    """Assigns MEDIUM tier to every prompt — disables routing signal."""

    def classify(self, messages: list[dict]) -> tuple[DifficultyTier, float]:
        return DifficultyTier.MEDIUM, 0.5

    def backend(self) -> str:
        return "fixed-medium"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mmlu(path: Path, n: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n is not None:
        rows = rows[:n]
    return rows


def load_done_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["source_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return done


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(response: str) -> tuple[str, bool]:
    """
    Extract A/B/C/D from first line of response.
    Returns (extracted, extraction_failed).
    """
    first_line = response.strip().split("\n")[0].strip().upper()
    for token in first_line.split():
        clean = token.strip("().:")
        if clean in VALID_ANSWERS:
            return clean, False
    return first_line[:1] if first_line else "", True


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _preflight_cascade() -> list[str]:
    issues = []
    if not settings.anthropic_api_key:
        issues.append("ANTHROPIC_API_KEY missing (claude-haiku-4-5)")
    if not settings.gemini_api_key:
        issues.append("GEMINI_API_KEY missing (gemini/gemini-2.5-flash + verifier)")
    if not settings.openai_api_key:
        issues.append("GROQ_API_KEY missing (groq/llama-3.3-70b-versatile frontier)")
    return issues


def preflight(mode: str) -> list[str]:
    if mode == "always-frontier":
        return [] if settings.groq_api_key else ["GROQ_API_KEY missing (groq/llama-3.3-70b-versatile)"]
    return _preflight_cascade()


# ---------------------------------------------------------------------------
# Cost estimation (dry-run)
# ---------------------------------------------------------------------------

def _cheap_cost_per_prompt() -> float:
    in_rate, out_rate = _COST_PER_1K["groq/llama-3.1-8b-instant"]
    return (_AVG_MMLU_INPUT_TOKENS * in_rate + _AVG_OUTPUT_TOKENS * out_rate) / 1000


def _frontier_cost_per_prompt() -> float:
    in_rate, out_rate = _COST_PER_1K["groq/llama-3.3-70b-versatile"]
    return (_AVG_MMLU_INPUT_TOKENS * in_rate + _AVG_OUTPUT_TOKENS * out_rate) / 1000


def estimate_cost(mode: str, rows: list[dict]) -> float:
    n = len(rows)
    if mode == "always-frontier":
        return n * _frontier_cost_per_prompt()
    if mode == "routing-only":
        tier_counts = Counter(r["difficulty_tier"] for r in rows)
        cheap_n = tier_counts.get("EASY", 0) + tier_counts.get("MEDIUM", 0)
        hard_n = tier_counts.get("HARD", 0)
        return cheap_n * _cheap_cost_per_prompt() + hard_n * _frontier_cost_per_prompt()
    if mode in ("cascade-only", "full"):
        # cheap call + verifier call (≈ cheap) + ~30% escalation to frontier
        escalation_rate = 0.30
        per_prompt = (
            _cheap_cost_per_prompt()       # cheap model
            + _cheap_cost_per_prompt()     # verifier call (similar size)
            + escalation_rate * _frontier_cost_per_prompt()
        )
        return n * per_prompt
    return 0.0


# ---------------------------------------------------------------------------
# JSONL append (crash-safe)
# ---------------------------------------------------------------------------

_write_lock = asyncio.Lock()


async def append_result(path: Path, result: BenchResult) -> None:
    async with _write_lock:
        with path.open("a") as f:
            f.write(json.dumps(asdict(result)) + "\n")


# ---------------------------------------------------------------------------
# Shared decision → result mapper
# ---------------------------------------------------------------------------

def _decision_to_result(
    row: dict,
    content: str,
    decision: RoutingDecision,
) -> BenchResult:
    extracted, failed = extract_answer(content)
    result = BenchResult(
        source_id=row["source_id"],
        subject=row["subject"],
        difficulty_tier=row["difficulty_tier"],
        correct_answer=row["correct_answer"],
        model_response=content,
        extracted_answer=extracted,
        extraction_failed=failed,
        is_correct=(not failed) and (extracted == row["correct_answer"]),
        request_id=decision.request_id,
        final_model=decision.final_model,
        cascade_path=decision.cascade_path,
        verifier_score=decision.verifier_score,
        estimated_cost_usd=decision.estimated_cost_usd,
        latency_ms=decision.latency_ms,
        escalated=decision.escalated,
        classifier_backend=decision.classifier_backend,
    )
    # Populate cheap answer for confusion matrix (when verifier ran)
    if decision.cheap_response:
        cheap_extracted, cheap_failed = extract_answer(decision.cheap_response)
        result.cheap_answer = cheap_extracted
        result.cheap_correct = (
            (not cheap_failed) and (cheap_extracted == row["correct_answer"])
        )
    return result


# ---------------------------------------------------------------------------
# Single-prompt runners
# ---------------------------------------------------------------------------

async def run_prompt_mode4(row: dict, sem: asyncio.Semaphore, results_path: Path) -> BenchResult:
    messages = [
        {"role": "system", "content": MMLU_SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]
    _FRONTIER = "groq/llama-3.3-70b-versatile"
    result = BenchResult(
        source_id=row["source_id"],
        subject=row["subject"],
        difficulty_tier=row["difficulty_tier"],
        correct_answer=row["correct_answer"],
        final_model=_FRONTIER,
        cascade_path=[_FRONTIER],
    )
    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await call_model(_FRONTIER, messages)
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.model_response = resp.content
            result.estimated_cost_usd = resp.estimated_cost_usd
            extracted, failed = extract_answer(resp.content)
            result.extracted_answer = extracted
            result.extraction_failed = failed
            result.is_correct = (not failed) and (extracted == row["correct_answer"])
        except ProviderError as e:
            result.latency_ms = (time.perf_counter() - t0) * 1000
            result.error = str(e)
            result.extraction_failed = True
    await append_result(results_path, result)
    return result


async def run_prompt_router(
    row: dict,
    router: Router,
    sem: asyncio.Semaphore,
    results_path: Path,
) -> BenchResult:
    """Shared runner for modes 1, 2, 3 — all use Router."""
    messages = [
        {"role": "system", "content": MMLU_SYSTEM_PROMPT},
        {"role": "user", "content": row["prompt"]},
    ]
    async with sem:
        try:
            content, decision = await router.route_async(
                messages,
                request_id=row["source_id"],
            )
            result = _decision_to_result(row, content, decision)
        except ProviderError as e:
            result = BenchResult(
                source_id=row["source_id"],
                subject=row["subject"],
                difficulty_tier=row["difficulty_tier"],
                correct_answer=row["correct_answer"],
                error=str(e),
                extraction_failed=True,
            )
    await append_result(results_path, result)
    return result


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def _make_router(mode: str) -> Router:
    if mode == "routing-only":
        return Router(
            classifier=get_classifier(settings.classifier),
            verifier_enabled=False,
        )
    if mode == "cascade-only":
        return Router(
            classifier=_AlwaysMediumClassifier(),
            verifier_enabled=True,
        )
    # "full"
    return Router(
        classifier=get_classifier(settings.classifier),
        verifier_enabled=True,
    )


def _progress_line(done: int, total: int, result: BenchResult, mode: str) -> str:
    status = "ERR" if result.error else ("OK" if result.is_correct else "WRONG")
    tier = f"[{result.difficulty_tier:<6s}]" if mode != "always-frontier" else ""
    return (
        f"  [{done:3d}/{total}] {status:5s}  {tier}  "
        f"{result.final_model:<25s}  {result.latency_ms:6.0f}ms  ${result.estimated_cost_usd:.5f}"
    )


async def run_mode(mode: str, rows: list[dict], results_path: Path, concurrency: int) -> list[BenchResult]:
    sem = asyncio.Semaphore(concurrency)

    if mode == "always-frontier":
        tasks = [run_prompt_mode4(row, sem, results_path) for row in rows]
    else:
        router = _make_router(mode)
        tasks = [run_prompt_router(row, router, sem, results_path) for row in rows]

    results = []
    done = 0
    total = len(tasks)
    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1
        print(_progress_line(done, total, result, mode))
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[BenchResult], mode_name: str) -> None:
    import statistics

    total = len(results)
    if total == 0:
        return

    errored = [r for r in results if r.error]
    failed_extraction = [r for r in results if r.extraction_failed and not r.error]
    evaluable = [r for r in results if not r.extraction_failed and not r.error]
    correct = [r for r in evaluable if r.is_correct]

    accuracy = len(correct) / len(evaluable) * 100 if evaluable else 0.0
    total_cost = sum(r.estimated_cost_usd for r in results)
    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    p50 = statistics.median(latencies) if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    escalated_n = sum(1 for r in results if r.escalated)

    print(f"\n{'='*62}")
    print(f"  Mode: {mode_name}")
    print(f"{'='*62}")
    print(f"  Prompts total:        {total}")
    print(f"  Errors:               {len(errored)}")
    print(f"  Extraction failures:  {len(failed_extraction)}  ({len(failed_extraction)/total*100:.1f}%)")
    print(f"  Evaluable:            {len(evaluable)}")
    print(f"  Accuracy:             {accuracy:.1f}%  ({len(correct)}/{len(evaluable)})")
    print(f"  Total cost:           ${total_cost:.4f}")
    print(f"  p50 latency:          {p50:.0f}ms")
    print(f"  p95 latency:          {p95:.0f}ms")
    if escalated_n > 0:
        print(f"  Escalation rate:      {escalated_n}/{total}  ({escalated_n/total*100:.1f}%)")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartRoute MMLU Benchmark")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--mode",
        choices=["routing-only", "cascade-only", "full", "always-frontier", "1", "2", "3", "4"],
        help="Single benchmark mode to run",
    )
    mode_group.add_argument(
        "--all-modes",
        action="store_true",
        help="Run all 4 modes sequentially",
    )

    parser.add_argument("--prompts", type=int, default=None,
                        help="Number of prompts (default: all 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without calling APIs")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                        help="Directory for result JSONL files")
    parser.add_argument("--concurrency", type=int, default=settings.benchmark_concurrency,
                        help=f"Max concurrent calls (default: {settings.benchmark_concurrency})")
    return parser.parse_args()


_MODE_ALIASES = {"1": "routing-only", "2": "cascade-only", "3": "full", "4": "always-frontier"}


def normalise_mode(s: str) -> str:
    return _MODE_ALIASES.get(s, s)


async def run_single(
    mode: str,
    rows: list[dict],
    output_dir: Path,
    concurrency: int,
    dry_run: bool,
) -> None:
    est = estimate_cost(mode, rows)
    if dry_run:
        print(f"\nDry-run  mode='{mode}'  n={len(rows)}")
        print(f"  Estimated cost: ${est:.4f}  (ceiling: ${settings.benchmark_cost_ceiling_usd:.2f})")
        if est > settings.benchmark_cost_ceiling_usd:
            print("  WARNING: Estimated cost exceeds ceiling — confirm before running.")
        return

    # Preflight
    issues = preflight(mode)
    if issues:
        print(f"WARNING ({mode}): API key issues:")
        for issue in issues:
            print(f"  - {issue}")

    # Cost ceiling
    if est > settings.benchmark_cost_ceiling_usd:
        print(f"\nWARNING: Estimated cost ${est:.4f} exceeds ceiling ${settings.benchmark_cost_ceiling_usd:.2f}")
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    # Output path
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_slug = mode.replace("-", "_")
    results_path = output_dir / f"run_{mode_slug}_{timestamp}.jsonl"

    # Resume
    done_ids = load_done_ids(results_path)
    pending = [r for r in rows if r["source_id"] not in done_ids]
    if done_ids:
        print(f"  Resuming: {len(done_ids)} done, {len(pending)} remaining")

    print(f"\nMode: {mode}  |  Prompts: {len(pending)}  |  Concurrency: {concurrency}")
    print(f"Output: {results_path}\n")

    results = await run_mode(mode, pending, results_path, concurrency)
    print_summary(results, mode)


async def main() -> None:
    args = parse_args()

    if not DATA_PATH.exists():
        print(f"ERROR: MMLU data not found at {DATA_PATH}")
        print("Run: uv run bench/data/mmlu_difficulty.py")
        sys.exit(1)

    rows = load_mmlu(DATA_PATH, n=args.prompts)
    print(f"Loaded {len(rows)} MMLU prompts")

    modes = ALL_MODES if args.all_modes else [normalise_mode(args.mode)]

    for mode in modes:
        if args.all_modes:
            print(f"\n{'#'*62}")
            print(f"  Running mode: {mode}")
            print(f"{'#'*62}")
        await run_single(mode, rows, args.output_dir, args.concurrency, args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())
