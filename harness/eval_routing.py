#!/usr/bin/env python
"""
Does the classifier route each kind of request to the tier it belongs on?

    uv run --extra ml harness/eval_routing.py
    uv run --extra ml harness/eval_routing.py --lambda-wrong 0.0005 --triage heuristic

Costs nothing and calls no models: decide() is pure, so the whole corpus can be routed
offline. Each question declares its own `expected_path`, so no hand-labelling is
involved at evaluation time.

Read the safety row, not just the accuracy row. Overall accuracy is dominated by
lookup recall, which is deliberately suppressed — layer 1 only fires when its measured
precision clears the bar the catalog economics set. The number that matters is how
much genuinely hard work gets dumped on the cheapest model.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

PATHS = ("cheap", "middle", "decompose", "frontier")


def build_triage(kind: str):
    from smartroute.decision import EmbeddingTriage, HeuristicTriage

    return HeuristicTriage() if kind == "heuristic" else EmbeddingTriage()


def evaluate(triage_kind: str, lambda_wrong: float, n: int | None, dataset_name: str = "synthetic") -> dict:
    """
    Route every question in the corpus and score it against its expected destination.

    Grouped by `expected_path` rather than by dataset: the synthetic corpus labels each
    item individually, so one corpus carries all three destinations.
    """
    from harness.corpora import get_dataset
    from smartroute.decision import DecisionLayer

    triage = build_triage(triage_kind)
    layer = DecisionLayer(triage=triage, lambda_wrong_usd=lambda_wrong)

    examples = get_dataset(dataset_name).load(n)
    by_expected: dict[str, list] = collections.defaultdict(list)
    for example in examples:
        by_expected[example.expected_path].append(example)

    rows, total_right = [], 0
    cheap_leakage = {}

    for expected, group in sorted(by_expected.items()):
        counts = collections.Counter(layer.decide(e.prompt).path for e in group)
        size = len(group)
        right = counts[expected]
        total_right += right
        rows.append({
            "expected": expected,
            "categories": sorted({e.metadata.get("category", "") for e in group}),
            "n": size,
            "distribution": {p: counts[p] / size for p in PATHS},
            "routed_right": right / size,
        })
        # "leakage" = work that needed more than the cheapest model but got it anyway
        if expected != "cheap":
            cheap_leakage[expected] = counts["cheap"] / size

    return {
        "triage": triage.backend(),
        "dataset": dataset_name,
        "lambda_wrong_usd": lambda_wrong,
        "lookup_threshold": layer._lookup_threshold,
        "decompose_threshold": layer._decompose_threshold,
        "required_lookup_precision": layer.required_lookup_precision(),
        "groups": rows,
        "overall_routed_right": total_right / len(examples),
        "cheap_leakage": cheap_leakage,
    }


def render(result: dict) -> None:
    print(f"\ntriage={result['triage']}  lambda=${result['lambda_wrong_usd']:.4f}  "
          f"lookup>={result['lookup_threshold']} (needs precision "
          f"{result['required_lookup_precision']:.1%})  decompose>={result['decompose_threshold']}")
    header = f"{'want (n)':28s}" + "".join(f"{p:>10s}" for p in PATHS) + f" | {'ROUTED RIGHT':>12s}"
    print(header)
    print("-" * len(header))
    for row in result["groups"]:
        label = f"{row['expected']} ({row['n']})"
        cells = "".join(f"{100 * row['distribution'][p]:9.1f}%" for p in PATHS)
        print(f"{label:28s}{cells} | {100 * row['routed_right']:11.1f}%")
    print(f"{'OVERALL':28s}{'':40s} | {100 * result['overall_routed_right']:11.1f}%")

    print("\nSAFETY — work that needed more than the cheapest model but was sent there anyway:")
    for name, rate in result["cheap_leakage"].items():
        flag = "  <-- " + ("ok" if rate < 0.05 else "PROBLEM")
        print(f"  {name:22s} {100 * rate:5.1f}%{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triage", choices=["embedding", "heuristic"], default="embedding")
    ap.add_argument("--lambda-wrong", type=float, default=0.001,
                    help="operator's cost of a wrong answer, USD (default 0.001)")
    ap.add_argument("--n", type=int, default=None, help="rows to evaluate (default all)")
    ap.add_argument("--dataset", type=str, default="synthetic")
    ap.add_argument("--compare", action="store_true", help="run both triage backends")
    ap.add_argument("--json", type=Path, help="also write results as JSON")
    args = ap.parse_args()

    kinds = ["heuristic", "embedding"] if args.compare else [args.triage]
    results = [evaluate(k, args.lambda_wrong, args.n, args.dataset) for k in kinds]
    for result in results:
        render(result)

    if len(results) == 2:
        before, after = results
        print("\n=== heuristic -> learned ===")
        print(f"  routing accuracy      {100*before['overall_routed_right']:5.1f}% -> "
              f"{100*after['overall_routed_right']:5.1f}%")
        for name in before["cheap_leakage"]:
            print(f"  want-{name} sent to cheap  {100*before['cheap_leakage'][name]:5.1f}% -> "
                  f"{100*after['cheap_leakage'][name]:5.1f}%")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
