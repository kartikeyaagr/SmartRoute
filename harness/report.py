"""
SmartRoute Benchmark Report Generator.

Reads result JSONLs from harness/results/, computes per-mode stats,
writes both machine-readable JSON and a markdown table.

Usage:
  uv run harness/report.py                          # auto-discover latest run per mode
  uv run harness/report.py --results-dir harness/results/
  uv run harness/report.py --files f1.jsonl f2.jsonl f3.jsonl f4.jsonl
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

RESULTS_DIR = Path(__file__).parent / "results"
_MODE_SLUG_ORDER = ["routing_only", "cascade_only", "full", "always_frontier"]
_MODE_DISPLAY = {
    "routing_only":    "Mode 1: routing-only",
    "cascade_only":    "Mode 2: cascade-only",
    "full":            "Mode 3: full SmartRoute",
    "always_frontier": "Mode 4: always-frontier",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModeStats:
    mode_slug: str
    n_total: int = 0
    n_errors: int = 0
    n_extraction_failed: int = 0
    n_evaluable: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    total_cost_usd: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    escalation_rate: float = 0.0
    extraction_failure_rate: float = 0.0
    # Confusion matrix — only populated for cascade modes (2, 3)
    verifier_pass_cheap_correct: int = 0    # verifier passed, cheap WAS correct
    verifier_pass_cheap_wrong: int = 0      # verifier passed, cheap was wrong (missed!)
    verifier_escalate_cheap_correct: int = 0  # escalated unnecessarily (cost waste)
    verifier_escalate_cheap_wrong: int = 0    # escalated correctly (caught bad answer)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_latest_files(results_dir: Path) -> dict[str, Path]:
    """
    For each mode slug, find the most recently modified JSONL file.
    Returns {mode_slug: path}.
    """
    found: dict[str, Path] = {}
    for path in sorted(results_dir.glob("run_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        for slug in _MODE_SLUG_ORDER:
            if slug in path.name and slug not in found:
                found[slug] = path
    return found


def infer_mode_slug(path: Path) -> str | None:
    name = path.stem  # e.g. run_full_20240101_120000
    for slug in _MODE_SLUG_ORDER:
        if slug in name:
            return slug
    return None


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def compute_stats(mode_slug: str, records: list[dict]) -> ModeStats:
    stats = ModeStats(mode_slug=mode_slug, n_total=len(records))

    latencies = []
    escalated_n = 0

    for r in records:
        if r.get("error"):
            stats.n_errors += 1
            continue
        if r.get("extraction_failed"):
            stats.n_extraction_failed += 1
            continue
        stats.n_evaluable += 1
        if r.get("is_correct"):
            stats.n_correct += 1
        stats.total_cost_usd += r.get("estimated_cost_usd", 0.0)
        lat = r.get("latency_ms", 0.0)
        if lat > 0:
            latencies.append(lat)
        if r.get("escalated"):
            escalated_n += 1

        # Confusion matrix
        verifier_score = r.get("verifier_score")
        cheap_correct = r.get("cheap_correct")
        if verifier_score is not None and cheap_correct is not None:
            escalated = r.get("escalated", False)
            if escalated:
                # verifier voted to escalate (score < threshold)
                if cheap_correct:
                    stats.verifier_escalate_cheap_correct += 1
                else:
                    stats.verifier_escalate_cheap_wrong += 1
            else:
                # verifier passed (score >= threshold)
                if cheap_correct:
                    stats.verifier_pass_cheap_correct += 1
                else:
                    stats.verifier_pass_cheap_wrong += 1

    # Also count errors toward total cost
    for r in records:
        if not r.get("error") and r.get("extraction_failed"):
            stats.total_cost_usd += r.get("estimated_cost_usd", 0.0)

    if stats.n_evaluable > 0:
        stats.accuracy = stats.n_correct / stats.n_evaluable * 100

    if latencies:
        stats.p50_latency_ms = statistics.median(latencies)
        stats.p95_latency_ms = sorted(latencies)[int(len(latencies) * 0.95)]

    if stats.n_total > 0:
        stats.escalation_rate = escalated_n / stats.n_total * 100
        stats.extraction_failure_rate = stats.n_extraction_failed / stats.n_total * 100

    return stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(mode_stats: list[ModeStats]) -> dict:
    """Build machine-readable JSON report."""
    frontier_stats = next((s for s in mode_stats if s.mode_slug == "always_frontier"), None)

    comparisons = {}
    if frontier_stats and frontier_stats.accuracy > 0:
        for s in mode_stats:
            if s.mode_slug == "always_frontier":
                continue
            quality_retention = s.accuracy / frontier_stats.accuracy if frontier_stats.accuracy else None
            cost_reduction = (
                (frontier_stats.total_cost_usd - s.total_cost_usd) / frontier_stats.total_cost_usd
                if frontier_stats.total_cost_usd > 0 else None
            )
            comparisons[s.mode_slug] = {
                "quality_retention": round(quality_retention * 100, 1) if quality_retention else None,
                "cost_reduction_pct": round(cost_reduction * 100, 1) if cost_reduction is not None else None,
            }

    return {
        "generated_at": datetime.now().isoformat(),
        "modes": {s.mode_slug: asdict(s) for s in mode_stats},
        "comparisons_vs_always_frontier": comparisons,
    }


def build_markdown(mode_stats: list[ModeStats], report: dict) -> str:
    lines = [
        "# SmartRoute Benchmark Results",
        "",
        f"_Generated: {report['generated_at']}_",
        "",
        "## Per-mode Summary",
        "",
        "| Mode | Accuracy | Cost ($) | p50 (ms) | p95 (ms) | Escalation | Extr. Fail |",
        "|------|----------|----------|----------|----------|------------|------------|",
    ]

    for s in mode_stats:
        display = _MODE_DISPLAY.get(s.mode_slug, s.mode_slug)
        lines.append(
            f"| {display} "
            f"| {s.accuracy:.1f}% ({s.n_correct}/{s.n_evaluable}) "
            f"| ${s.total_cost_usd:.4f} "
            f"| {s.p50_latency_ms:.0f} "
            f"| {s.p95_latency_ms:.0f} "
            f"| {s.escalation_rate:.1f}% "
            f"| {s.extraction_failure_rate:.1f}% |"
        )

    # Comparisons vs frontier
    comparisons = report.get("comparisons_vs_always_frontier", {})
    if comparisons:
        lines += [
            "",
            "## vs Always-Frontier Baseline",
            "",
            "| Mode | Quality Retention | Cost Reduction |",
            "|------|-------------------|----------------|",
        ]
        for slug, comp in comparisons.items():
            display = _MODE_DISPLAY.get(slug, slug)
            qr = f"{comp['quality_retention']}%" if comp["quality_retention"] is not None else "n/a"
            cr = f"{comp['cost_reduction_pct']}%" if comp["cost_reduction_pct"] is not None else "n/a"
            lines.append(f"| {display} | {qr} | {cr} |")

    # Verifier confusion matrices
    cascade_modes = [s for s in mode_stats if s.mode_slug in ("cascade_only", "full")]
    if cascade_modes:
        lines += ["", "## Verifier Confusion Matrix", ""]
        for s in cascade_modes:
            total_verifier = (
                s.verifier_pass_cheap_correct
                + s.verifier_pass_cheap_wrong
                + s.verifier_escalate_cheap_correct
                + s.verifier_escalate_cheap_wrong
            )
            if total_verifier == 0:
                continue
            display = _MODE_DISPLAY.get(s.mode_slug, s.mode_slug)
            lines += [
                f"### {display}",
                "",
                "| | Cheap correct | Cheap wrong |",
                "|---|---|---|",
                f"| Verifier passed  | {s.verifier_pass_cheap_correct} (good) | {s.verifier_pass_cheap_wrong} (missed!) |",
                f"| Verifier escalated | {s.verifier_escalate_cheap_correct} (wasted) | {s.verifier_escalate_cheap_wrong} (caught!) |",
                "",
            ]

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmartRoute Benchmark Report")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Directory containing result JSONL files")
    parser.add_argument("--files", type=Path, nargs="+",
                        help="Explicit list of result JSONL files (overrides --results-dir)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for report output (default: same as results-dir)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_dir

    # Discover or use explicit files
    if args.files:
        file_map: dict[str, Path] = {}
        for f in args.files:
            slug = infer_mode_slug(f)
            if slug:
                file_map[slug] = f
            else:
                print(f"WARNING: Could not infer mode from filename {f.name}, skipping")
    else:
        file_map = discover_latest_files(args.results_dir)

    if not file_map:
        print(f"No result files found in {args.results_dir}")
        print("Run benchmark first: uv run harness/run_benchmark.py --all-modes")
        sys.exit(1)

    print(f"Found {len(file_map)} result file(s):")
    mode_stats = []
    for slug in _MODE_SLUG_ORDER:
        if slug not in file_map:
            continue
        path = file_map[slug]
        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        print(f"  {slug:<20s} {len(records):4d} records  {path.name}")
        stats = compute_stats(slug, records)
        mode_stats.append(stats)

    if not mode_stats:
        print("No valid records found.")
        sys.exit(1)

    report = build_report(mode_stats)
    md = build_markdown(mode_stats, report)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"report_{timestamp}.json"
    md_path = output_dir / f"report_{timestamp}.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(md)

    print(f"\nReports written:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print(md)


if __name__ == "__main__":
    main()
