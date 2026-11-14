#!/usr/bin/env python
"""
Fit the learned answer checker used by ESCALATION_CHECKER=learned|both.

    # from your own graded traffic (recommended): JSONL of {"prompt", "response", "correct"}
    uv run --extra ml harness/train_checker.py --jsonl graded_middle_answers.jsonl

    # from RouterBench, using one model's recorded answers (a stand-in, see below)
    uv run --extra ml --with pandas harness/train_checker.py \
        --routerbench routerbench_5shot.pkl --model mistralai/mixtral-8x7b-chat

The checker is a classifier over MiniLM embeddings of (prompt, answer) -> P(correct).
It learns what *that model's* mistakes look like, so train it on answers from the middle
model you actually serve. A RouterBench-trained checker has never seen gpt-oss-120b and
its RouterBench AUC says nothing about how it transfers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from harness.routerbench import ENCODER, embed, fit_checker, load  # noqa: E402
from smartroute.escalation import DEFAULT_CHECKER_PATH, pair_features, tail  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--jsonl", type=Path)
    src.add_argument("--routerbench", type=Path)
    ap.add_argument("--model", default="mistralai/mixtral-8x7b-chat", help="RouterBench answer column")
    ap.add_argument("--tail-chars", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "harness" / "results" / ".cache")
    ap.add_argument("--out", type=Path, default=DEFAULT_CHECKER_PATH)
    args = ap.parse_args()

    if args.jsonl:
        rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
        prompts = [r["prompt"] for r in rows]
        responses = [r["response"] for r in rows]
        correct = np.array([bool(r["correct"]) for r in rows])
        source = f"jsonl:{args.jsonl.name}"
    else:
        df = load(args.routerbench, None, args.seed)
        prompts = df["text"].tolist()
        responses = [str(r) if r is not None else "" for r in df[f"{args.model}|model_response"]]
        correct = df[args.model].to_numpy(float) >= 0.5
        source = f"routerbench:{args.routerbench.name}:{args.model}"

    P = embed([tail(p, args.tail_chars) for p in prompts], args.cache_dir, "prompt")
    R = embed([tail(r, args.tail_chars) for r in responses], args.cache_dir, "resp-train")
    head, kind, val_auc = fit_checker(pair_features(P, R), correct, args.seed)

    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"head": head, "encoder": ENCODER, "tail_chars": args.tail_chars, "trained_on": source},
                args.out)
    print(f"{kind} checker, validation AUC {val_auc[kind]:.3f} ({val_auc}), {len(correct)} rows -> {args.out}")


if __name__ == "__main__":
    main()
