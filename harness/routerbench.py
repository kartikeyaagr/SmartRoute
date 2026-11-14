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
# Derived from the Jev tutorial's sample response (476 input tokens -> $0.000019992),
# NOT from the live model page. Pass --jev-price-per-mtok to use the current price.
JEV_PRICE_PER_MTOK = 0.042
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


def evaluate(
    df, ladder: dict[str, str], triage_kind: str, lambda_wrong: float | None, decompose_as: str,
    jev_price_per_mtok: float = JEV_PRICE_PER_MTOK, live_n: int = 0,
    decision_model: str = "~typesafe/jev-latest", concurrency: int = 8, seed: int = 42,
) -> dict:
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

    routed_tiers = arms["routed"]
    escalation_sim = simulate_escalation(df, ladder, routed_tiers, jev_price_per_mtok)
    escalation_live = (
        live_escalation(df, ladder, routed_tiers, live_n, decision_model, concurrency, seed)
        if live_n else None
    )

    return {
        "escalation_sim": escalation_sim,
        "escalation_live": escalation_live,
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


# ---------------------------------------------------------------------------
# Escalation: Jev checks the middle answer, frontier answers when it fails
# ---------------------------------------------------------------------------

_WRONG_BELOW = 0.5  # RouterBench scores are 0..1; some evals grade fractionally


def _jev_cost(df, ladder, price_per_mtok):
    resp = df[f"{ladder['middle']}|model_response"].fillna("").astype(str)
    tokens = (df["text"].str.len() + resp.str.len()) / 4.0
    return tokens * price_per_mtok / 1_000_000


def _escalated_arm(df, ladder, tiers, escalate, jev_cost):
    """Accuracy/cost when middle rows in `escalate` are re-answered by frontier."""
    mid, front = ladder["middle"], ladder["frontier"]
    scores, costs = [], []
    for i, t in enumerate(tiers):
        s = float(df.at[i, ladder[t]])
        c = float(df.at[i, f"{ladder[t]}|total_cost"])
        if t == "middle":
            e = escalate[i]
            s = (1 - e) * s + e * float(df.at[i, front])
            c = c + float(jev_cost.iat[i]) + e * float(df.at[i, f"{front}|total_cost"])
        scores.append(s)
        costs.append(c)
    return sum(scores) / len(scores), sum(costs) / len(costs)


def simulate_escalation(df, ladder, tiers, price_per_mtok) -> dict:
    """
    How good does the checker have to be? Sweep detection rate (P(escalate | middle
    wrong)) against false-alarm rate (P(escalate | middle right)). The perfect checker
    (1.0, 0.0) is the ceiling any real Jev threshold can reach.
    """
    jev = _jev_cost(df, ladder, price_per_mtok)
    wrong = [float(df.at[i, ladder["middle"]]) < _WRONG_BELOW for i in range(len(df))]
    grid = {}
    for detect in (1.0, 0.9, 0.8, 0.7, 0.5):
        for false_alarm in (0.0, 0.1, 0.2, 0.3, 0.5):
            esc = [detect if w else false_alarm for w in wrong]
            acc, cost = _escalated_arm(df, ladder, tiers, esc, jev)
            grid[f"{detect:.1f}/{false_alarm:.1f}"] = {"accuracy": acc, "cost_per_query": cost}
    middle_rows = [i for i, t in enumerate(tiers) if t == "middle"]
    return {
        "jev_price_per_mtok": price_per_mtok,
        "jev_cost_per_check": float(jev.iloc[middle_rows].mean()) if middle_rows else 0.0,
        "middle_wrong_rate": (sum(wrong[i] for i in middle_rows) / len(middle_rows)) if middle_rows else 0.0,
        "grid": grid,
    }


async def _live_checks(df, ladder, rows, model, concurrency):
    import asyncio
    import os

    from smartroute.escalation import JevVerifier

    verifier = JevVerifier(os.environ.get("OPENROUTER_API_KEY", ""), threshold=0.5, model=model)
    sem = asyncio.Semaphore(concurrency)
    col = f"{ladder['middle']}|model_response"

    async def one(i):
        async with sem:
            r = await verifier.check(df.at[i, "text"], str(df.at[i, col] or ""))
            return i, r.p_correct, r.cost_usd

    return await asyncio.gather(*(one(i) for i in rows))


def live_escalation(df, ladder, tiers, n, model, concurrency, seed) -> dict:
    """Real Jev calls on recorded middle answers. Costs money; needs OPENROUTER_API_KEY."""
    import asyncio
    import random

    middle_rows = [i for i, t in enumerate(tiers) if t == "middle"]
    rows = sorted(random.Random(seed).sample(middle_rows, min(n, len(middle_rows))))
    checks = asyncio.run(_live_checks(df, ladder, rows, model, concurrency))
    ok = [(i, p, c) for i, p, c in checks if p is not None]
    if not ok:
        raise SystemExit("every Jev check failed — is OPENROUTER_API_KEY set?")

    mid, front = ladder["middle"], ladder["frontier"]
    sub = df.loc[[i for i, _, _ in ok]].reset_index(drop=True)
    sub_tiers = ["middle"] * len(sub)
    jev = __import__("pandas").Series([c for _, _, c in ok])  # actual billed cost
    wrong = [float(sub.at[j, mid]) < _WRONG_BELOW for j in range(len(sub))]
    probs = [p for _, p, _ in ok]

    base_acc = float(sub[mid].mean())
    base_cost = float(sub[f"{mid}|total_cost"].mean())
    sweep = {}
    for th in (0.3, 0.5, 0.6, 0.7, 0.78, 0.85, 0.9, 0.95):
        esc = [1.0 if p < th else 0.0 for p in probs]
        acc, cost = _escalated_arm(sub, ladder, sub_tiers, esc, jev)
        n_wrong = sum(wrong) or 1
        n_right = (len(wrong) - sum(wrong)) or 1
        sweep[f"{th:.2f}"] = {
            "accuracy": acc,
            "cost_per_query": cost,
            "escalation_rate": sum(esc) / len(esc),
            "detection_rate": sum(e for e, w in zip(esc, wrong) if w) / n_wrong,
            "false_alarm_rate": sum(e for e, w in zip(esc, wrong) if not w) / n_right,
        }
    return {
        "n_checked": len(ok),
        "n_failed": len(checks) - len(ok),
        "jev_cost_per_check": float(jev.mean()),
        "middle_accuracy": base_acc,
        "middle_cost_per_query": base_cost,
        "frontier_accuracy": float(sub[front].mean()),
        "frontier_cost_per_query": float(sub[f"{front}|total_cost"].mean()),
        "threshold_sweep": sweep,
    }


def render_escalation(r: dict) -> None:
    sim = r.get("escalation_sim")
    if sim:
        print(f"\nEscalation ceiling (simulated checker) — middle wrong on "
              f"{sim['middle_wrong_rate']:.1%} of middle-routed prompts; "
              f"Jev ≈ ${sim['jev_cost_per_check']:.6f}/check at ${sim['jev_price_per_mtok']}/M (estimate)")
        base = r["arms"]["routed"]
        print(f"routed (no escalation): {base['accuracy']:.1%} @ ${base['cost_per_query']:.6f}")
        print(f"{'detect/false-alarm':<20}{'accuracy':>10}{'$/query':>12}")
        for k, v in sim["grid"].items():
            print(f"{k:<20}{v['accuracy']:>10.1%}{v['cost_per_query']:>12.6f}")
    live = r.get("escalation_live")
    if live:
        print(f"\nLive Jev checks — n={live['n_checked']} (failed {live['n_failed']}), "
              f"${live['jev_cost_per_check']:.6f}/check")
        print(f"same rows: middle {live['middle_accuracy']:.1%} @ ${live['middle_cost_per_query']:.6f}, "
              f"frontier {live['frontier_accuracy']:.1%} @ ${live['frontier_cost_per_query']:.6f}")
        print(f"{'threshold':<11}{'accuracy':>10}{'$/query':>12}{'escalate':>10}{'detect':>8}{'false+':>8}")
        for th, v in live["threshold_sweep"].items():
            print(f"{th:<11}{v['accuracy']:>10.1%}{v['cost_per_query']:>12.6f}"
                  f"{v['escalation_rate']:>10.1%}{v['detection_rate']:>8.1%}{v['false_alarm_rate']:>8.1%}")


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
    ap.add_argument("--jev-price-per-mtok", type=float, default=JEV_PRICE_PER_MTOK,
                    help="Jev input price for the simulated checker (default: tutorial-derived estimate)")
    ap.add_argument("--jev-verify", type=int, default=0, metavar="N",
                    help="also run N live Jev checks on recorded middle answers (costs money; "
                         "needs OPENROUTER_API_KEY)")
    ap.add_argument("--decision-model", default="~typesafe/jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "harness" / "results" / "routerbench.json")
    args = ap.parse_args()

    ladder = {"cheap": args.cheap, "middle": args.middle, "frontier": args.frontier}
    df = load(args.data, args.n, args.seed)
    result = evaluate(
        df, ladder, args.triage, args.lambda_wrong, args.decompose_as,
        jev_price_per_mtok=args.jev_price_per_mtok, live_n=args.jev_verify,
        decision_model=args.decision_model, concurrency=args.concurrency, seed=args.seed,
    )
    render(result)
    render_escalation(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
