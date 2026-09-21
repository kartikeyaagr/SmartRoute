#!/usr/bin/env python
"""
Materialise the evaluation corpora to JSONL under harness/data/.

Downloads once, writes a fixed sample, and is never needed again — runs read the
JSONL so they are deterministic and work offline. Mirrors the approach already used
by harness/data/mmlu_difficulty.py for MMLU.

    uv run --extra bench harness/build_corpora.py
    uv run --extra bench harness/build_corpora.py --n 500 --only triviaqa

Dedup is on the (question, answer) pair rather than a hash of the question alone:
the same question appears under several HotpotQA/TriviaQA sources with different
gold spans, and dropping by question alone silently discards valid variants.
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# TriviaQA questions are already short lookups, but a handful are long multi-clause
# items that are not "google substitutes". Cap length so the LOOKUP class stays clean —
# it is the training signal for layer 1, and a noisy positive class is worse than a
# smaller one.
_MAX_LOOKUP_CHARS = 200


def _dedup(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["prompt"].strip(), str(r["answer"]).strip())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def build_triviaqa(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    print(f"  downloading TriviaQA (rc.nocontext)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    rows = []
    for r in ds:
        q = r["question"].strip()
        if not q or len(q) > _MAX_LOOKUP_CHARS:
            continue
        answer = r["answer"]
        rows.append({
            "source_id": f"triviaqa:{r['question_id']}",
            "prompt": q,
            "answer": answer["value"],
            "aliases": sorted(set(answer.get("aliases") or []))[:20],
        })
    rows = _dedup(rows)
    print(f"  {len(rows)} usable rows after dedup/length filter")
    return random.Random(seed).sample(rows, min(n, len(rows)))


def build_hotpotqa(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    print(f"  downloading HotpotQA (distractor)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    for r in ds:
        q = r["question"].strip()
        if not q:
            continue
        rows.append({
            "source_id": f"hotpotqa:{r['id']}",
            "prompt": q,
            "answer": r["answer"],
            "type": r.get("type"),      # bridge | comparison
            "level": r.get("level"),    # easy | medium | hard
            # how many distinct documents the gold reasoning path touches: the
            # ground-truth "how many sub-tasks would this need" signal
            "n_hops": len({t for t in (r.get("supporting_facts") or {}).get("title", [])}),
        })
    rows = _dedup(rows)
    print(f"  {len(rows)} usable rows after dedup")
    return random.Random(seed).sample(rows, min(n, len(rows)))


BUILDERS = {"triviaqa": build_triviaqa, "hotpotqa": build_hotpotqa}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500, help="rows per dataset (default 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", choices=sorted(BUILDERS), help="build just one dataset")
    ap.add_argument("--force", action="store_true", help="rebuild even if the file exists")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    targets = [args.only] if args.only else sorted(BUILDERS)

    for name in targets:
        out = DATA_DIR / f"{name}_{args.n}.jsonl"
        if out.exists() and not args.force:
            print(f"{name}: {out.name} already exists — skipping (use --force to rebuild)")
            continue
        print(f"{name}:")
        rows = BUILDERS[name](args.n, args.seed)
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows)} rows -> {out}")

    print("\nMMLU is already materialised at harness/data/mmlu_500.jsonl")


if __name__ == "__main__":
    main()
