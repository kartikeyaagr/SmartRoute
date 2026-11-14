#!/usr/bin/env python
"""
Replay the two-layer decision router over RouterBench, offline.

    uv run --extra ml --with pandas harness/routerbench.py --data routerbench_0shot.pkl

RouterBench (Hu et al., 2024; huggingface.co/datasets/withmartian/routerbench) records,
for ~36k prompts, every one of 11 models' response, graded score (0..1) and dollar
cost. That turns routing evaluation into a lookup: run decide() on the prompt, map the
chosen tier to a RouterBench model, and read off that model's recorded score and cost.
No model is called, so the whole benchmark costs nothing.

What this does NOT measure
--------------------------
* Our catalog's models. RouterBench predates them, so tiers are mapped onto the
  closest RouterBench ladder (default mistral-7b -> mixtral-8x7b -> gpt-4). This
  tests the *routing decisions*, priced and graded with RouterBench's models.
* The decompose path. Layer 2 issues live sub-calls that RouterBench never recorded,
  so decompose-routed prompts are scored as the frontier model (`--decompose-as`).
  The count is reported so you can see how much this proxy matters.
* Leakage: the shipped triage heads were trained with MMLU rows (as "not a lookup"),
  and RouterBench contains MMLU. The mmlu rows are reported separately for that reason.

Security: RouterBench ships only as a pickle, and unpickling runs code. The loader
unpickles with an allowlist: any global outside pandas/numpy/builtins aborts the load.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

DEFAULT_LADDER = {
    "cheap": "mistralai/mistral-7b-chat",
    "middle": "mistralai/mixtral-8x7b-chat",
    "frontier": "gpt-4-1106-preview",
}
_SAFE_MODULES = frozenset({"pandas", "numpy", "builtins", "copyreg", "_codecs"})


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class _RestrictedUnpickler(pickle.Unpickler):
    """Resolve only pandas/numpy/builtins globals; anything else aborts the load."""

    def find_class(self, module: str, name: str):
        if module.split(".")[0] in _SAFE_MODULES:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name}")


def _user_text(raw) -> str:
    """RouterBench prompts are str(list_of_turns); routing reads the last user turn."""
    if isinstance(raw, str) and raw.startswith("["):
        try:
            turns = ast.literal_eval(raw)
            if isinstance(turns, list) and turns:
                return "\n".join(str(t) for t in turns)
        except (ValueError, SyntaxError):
            pass
    return str(raw)


def load(path: Path, n: int | None, seed: int):
    import pandas as pd

    with path.open("rb") as f:
        df = _RestrictedUnpickler(f).load()
    if not isinstance(df, pd.DataFrame):
        raise SystemExit(f"{path} did not contain a DataFrame")
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    df = df.reset_index(drop=True)
    df["text"] = df["prompt"].map(_user_text)
    return df


# ---------------------------------------------------------------------------
# Batched triage — same encoder and heads as EmbeddingTriage, 100x fewer calls
# ---------------------------------------------------------------------------

class PrecomputedTriage:
    def __init__(self, texts: list[str], kind: str):
        from smartroute.decision import EmbeddingTriage, HeuristicTriage, TriageSignals

        self._signals: dict[str, TriageSignals] = {}
        if kind == "heuristic":
            h = HeuristicTriage()
            self._signals = {t: h.signals(t) for t in texts}
            self.calibration = {}
            self._backend = "heuristic"
            return

        inner = EmbeddingTriage()
        if inner._heads is None:
            raise SystemExit("no trained triage model; run harness/train_triage.py or pass --triage heuristic")
        self.calibration = inner.calibration
        inner._encode("warmup")
        x = inner._encoder.encode(
            texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True
        )
        p_lookup = inner._heads["lookup"].predict_proba(x)[:, 1]
        p_dec = inner._heads["decompose"].predict_proba(x)[:, 1]
        self._signals = {
            t: TriageSignals(p_lookup=float(a), p_decompose=float(b), backend="embedding")
            for t, a, b in zip(texts, p_lookup, p_dec)
        }
        self._backend = "embedding"

    def signals(self, prompt: str):
        return self._signals[prompt]

    def backend(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _group(eval_name: str) -> str:
    if eval_name.startswith("mmlu"):
        return "mmlu"
    if eval_name.startswith(("chinese", "Chinese")):
        return "chinese"
    if eval_name.startswith("mtbench"):
        return "mtbench"
    return eval_name


def evaluate(df, ladder: dict[str, str], triage_kind: str, lambda_wrong: float | None, decompose_as: str) -> dict:
    from smartroute.decision import DecisionLayer

    texts = df["text"].tolist()
    triage = PrecomputedTriage(list(dict.fromkeys(texts)), triage_kind)
    layer = DecisionLayer(triage=triage, lambda_wrong_usd=lambda_wrong)

    paths = [layer.decide(t).path for t in texts]
    tier_for = lambda p: decompose_as if p == "decompose" else p  # noqa: E731
    df = df.assign(path=paths, group=df["eval_name"].map(_group))

    def arm_cols(tiers):
        score = [float(df.at[i, ladder[t]]) for i, t in enumerate(tiers)]
        cost = [float(df.at[i, f"{ladder[t]}|total_cost"]) for i, t in enumerate(tiers)]
        return score, cost

    arms = {
        "always-cheap": ["cheap"] * len(df),
        "always-middle": ["middle"] * len(df),
        "always-frontier": ["frontier"] * len(df),
        "routed": [tier_for(p) for p in paths],
    }
    # Oracle: the cheapest ladder model that scores > 0, else the cheap one.
    oracle = []
    for i in range(len(df)):
        pick = "cheap"
        for t in ("cheap", "middle", "frontier"):
            if float(df.at[i, ladder[t]]) > 0:
                pick = t
                break
        oracle.append(pick)
    arms["oracle (ladder)"] = oracle

    results = {}
    for name, tiers in arms.items():
        score, cost = arm_cols(tiers)
        d = df.assign(_s=score, _c=cost)
        per_group = d.groupby("group").agg(score=("_s", "mean"), cost=("_c", "mean"), n=("_s", "size"))
        results[name] = {
            "accuracy": float(d["_s"].mean()),
            "macro_accuracy": float(per_group["score"].mean()),
            "cost_per_query": float(d["_c"].mean()),
            "total_cost": float(d["_c"].sum()),
            "by_group": per_group.round(6).to_dict(orient="index"),
        }

    return {
        "n": len(df),
        "ladder": ladder,
        "triage": triage.backend(),
        "lookup_threshold": layer._lookup_threshold,
        "required_lookup_precision": layer.required_lookup_precision(),
        "path_counts": dict(collections.Counter(paths)),
        "path_by_group": df.groupby("group")["path"].value_counts().unstack(fill_value=0).to_dict(orient="index"),
        "decompose_scored_as": decompose_as,
        "arms": results,
    }


def render(r: dict) -> None:
    print(f"\nRouterBench replay — n={r['n']}, triage={r['triage']}, "
          f"lookup threshold={r['lookup_threshold']:.2f} "
          f"(needs precision {r['required_lookup_precision']:.1%})")
    print("ladder: " + " -> ".join(f"{k}={v}" for k, v in r["ladder"].items()))
    print("paths:  " + ", ".join(f"{k}={v}" for k, v in sorted(r["path_counts"].items()))
          + f"   (decompose scored as {r['decompose_scored_as']})\n")

    base = r["arms"]["always-cheap"]["cost_per_query"]
    print(f"{'arm':<18}{'accuracy':>10}{'macro acc':>11}{'$/query':>12}{'vs cheap':>10}")
    for name, a in r["arms"].items():
        print(f"{name:<18}{a['accuracy']:>10.1%}{a['macro_accuracy']:>11.1%}"
              f"{a['cost_per_query']:>12.6f}{a['cost_per_query'] / base:>9.2f}x")

    groups = sorted(r["arms"]["routed"]["by_group"], key=lambda g: -r["arms"]["routed"]["by_group"][g]["n"])
    print(f"\n{'group':<22}{'n':>6}  {'cheap':>7}{'middle':>8}{'front':>8}{'routed':>8}   paths")
    for g in groups:
        row = [r["arms"][a]["by_group"][g]["score"] for a in ("always-cheap", "always-middle", "always-frontier", "routed")]
        pc = r["path_by_group"].get(g, {})
        print(f"{g:<22}{r['arms']['routed']['by_group'][g]['n']:>6}  "
              + "".join(f"{x:>8.1%}" if i else f"{x:>7.1%}" for i, x in enumerate(row))
              + "   " + " ".join(f"{k[0]}={v}" for k, v in pc.items() if v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="routerbench_0shot.pkl")
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triage", choices=("embedding", "heuristic"), default="embedding")
    ap.add_argument("--lambda-wrong", type=float, default=None, help="default: catalog value")
    ap.add_argument("--cheap", default=DEFAULT_LADDER["cheap"])
    ap.add_argument("--middle", default=DEFAULT_LADDER["middle"])
    ap.add_argument("--frontier", default=DEFAULT_LADDER["frontier"])
    ap.add_argument("--decompose-as", choices=("cheap", "middle", "frontier"), default="frontier")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "harness" / "results" / "routerbench.json")
    args = ap.parse_args()

    ladder = {"cheap": args.cheap, "middle": args.middle, "frontier": args.frontier}
    df = load(args.data, args.n, args.seed)
    result = evaluate(df, ladder, args.triage, args.lambda_wrong, args.decompose_as)
    render(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
