#!/usr/bin/env python
"""
Fit the two layer-1 triage heads.

    uv run --extra ml python -m harness.train_triage

Why this exists
---------------
The regex heuristic sends 41.6% of HotpotQA's multi-hop questions to the *cheapest*
model, because a 2-hop question opens exactly like a lookup:

    lookup   "In what year was Harvard founded?"
    2-hop    "In what year was the university where Tokarev was a professor founded?"

Both start "In what year"; only the second hides a second retrieval. No pattern over
surface forms separates them, which is why the heads are learned over sentence
embeddings instead.

Labels are free — each item's expected_path *is* the routing label:

    cheap     -> lookup=1, decompose=0
    middle    -> lookup=0, decompose=0
    decompose -> lookup=0, decompose=1

The head is deliberately tiny (logistic regression on frozen embeddings). It has to
run on every request for free, and there are only a few hundred training rows; anything
with more capacity would memorise them.
"""

import argparse
import json
from pathlib import Path

MODEL_OUT = Path(__file__).resolve().parents[1] / "src" / "smartroute" / "data" / "triage.joblib"
ENCODER = "all-MiniLM-L6-v2"


def load_labelled(corpus: Path | None) -> tuple[list[str], list[int], list[int], list[str]]:
    """Prompts, both head labels (from each item's expected_path), and source ids."""
    from harness.corpora import SyntheticDataset

    try:
        examples = SyntheticDataset(corpus).load()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    return (
        [ex.prompt for ex in examples],
        [int(ex.expected_path == "cheap") for ex in examples],
        [int(ex.expected_path == "decompose") for ex in examples],
        [ex.source_id for ex in examples],
    )


def _calibration_curve(head, x_test, y_test) -> list[dict]:
    """
    Measured precision/recall at each candidate threshold.

    Persisted with the model because the decision layer derives its operating
    threshold from the economics — required precision is a function of catalog prices
    and lambda — and needs to know which threshold actually delivers that precision.
    A hardcoded 0.75 is a guess; this is a measurement.
    """
    probabilities = head.predict_proba(x_test)[:, 1]
    curve = []
    for threshold in [round(0.50 + 0.05 * i, 2) for i in range(10)]:
        predicted = probabilities >= threshold
        tp = int((predicted & (y_test == 1)).sum())
        fp = int((predicted & (y_test == 0)).sum())
        fn = int(((~predicted) & (y_test == 1)).sum())
        if tp + fp == 0:
            continue
        curve.append({
            "threshold": threshold,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        })
    return curve


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=MODEL_OUT)
    ap.add_argument("--corpus", type=Path, default=None, help="corpus JSONL (default: the synthetic 200)")
    args = ap.parse_args()

    import joblib
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    prompts, lookup, decompose, ids = load_labelled(args.corpus)
    print(f"{len(prompts)} labelled prompts")
    print(f"  lookup positives: {sum(lookup)}   decompose positives: {sum(decompose)}")

    print(f"encoding with {ENCODER}...")
    encoder = SentenceTransformer(ENCODER)
    features = encoder.encode(prompts, normalize_embeddings=True, show_progress_bar=False)

    import numpy as np

    # One split shared by both heads, so a single set of held-out ids can be recorded.
    # Benchmark routing accuracy measured on rows the heads trained on is memorisation,
    # not generalisation, and has to be reported separately.
    idx = np.arange(len(prompts))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=np.array(decompose)
    )
    held_out_ids = sorted(ids[i] for i in test_idx)

    heads, report, calibration = {}, {}, {}
    for name, labels in (("lookup", lookup), ("decompose", decompose)):
        labels = np.array(labels)
        x_train, x_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        # balanced: the negative class is 2x the positive one for both heads, and an
        # unbalanced fit would buy accuracy by simply never predicting the positive.
        head = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)
        head.fit(x_train, y_train)
        heads[name] = head

        print(f"\n=== {name} head ===")
        print(classification_report(y_test, head.predict(x_test), target_names=[f"not-{name}", name]))
        report[name] = classification_report(
            y_test, head.predict(x_test), target_names=[f"not-{name}", name], output_dict=True
        )
        calibration[name] = _calibration_curve(head, x_test, y_test)

        print(f"  {'threshold':>9s} {'precision':>10s} {'recall':>8s}")
        for point in calibration[name]:
            print(f"  {point['threshold']:9.2f} {point['precision']:10.2f} {point['recall']:8.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "heads": heads,
        "calibration": calibration,
        "encoder": ENCODER,
        "held_out_ids": held_out_ids,
        "trained_on": str(args.corpus or "synthetic"),
    }, args.out)
    print(f"\nheld-out ids recorded: {len(held_out_ids)} of {len(prompts)}")
    print(f"\nwrote {args.out}")
    (args.out.parent / "triage_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out.parent / 'triage_report.json'}")


if __name__ == "__main__":
    main()
