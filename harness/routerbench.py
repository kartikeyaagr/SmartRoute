#!/usr/bin/env python
"""
Replay the two-layer decision router — with and without answer checking — over
RouterBench, offline, and score it with the paper's AIQ metric.

    uv run --extra ml --with pandas harness/routerbench.py --data routerbench_5shot.pkl

RouterBench (Hu et al., 2024; huggingface.co/datasets/withmartian/routerbench) records,
for ~36k prompts, 11 models' responses, graded scores (0..1) and dollar costs. Routing
evaluation becomes a lookup: decide() picks a tier, the tier maps to a RouterBench model,
and that model's recorded score and cost are read off. No model is called.

Protocol (matches the paper's predictive-router setup)
------------------------------------------------------
Prompts are split 70/30 at random. The learned checker trains on the 70%; every number
reported — every arm, curve and AIQ — is on the held-out 30%.

Routers compared
----------------
  zero (ladder)       best mix of the 3 ladder models, no per-prompt routing
  zero (all 11)       the paper's Zero router: hull of all 11 models
  decision            the two-layer router, swept over lambda_wrong
  decision+learned    + learned answer checker escalating middle -> frontier,
                      swept over lambda x checker threshold
  decision+perfect    + a checker that is never wrong (the ceiling for any checker)

AIQ = mean height of a router's non-decreasing convex hull of (cost, accuracy) points
over the ladder's cost range [always-cheap, always-frontier]. Every router may fall back
to always-cheap/always-frontier, so AIQ(router) >= AIQ(zero ladder) by construction:
equal means the router adds nothing over picking a fixed model.

What this does NOT measure
--------------------------
* Our catalog's models. Tiers map onto RouterBench models, so this tests routing
  decisions, graded and priced with RouterBench's models.
* The decompose path: its live sub-calls were never recorded, so decompose-routed
  prompts are scored as the frontier model.
* Embeddings see only the last --tail-chars characters of each prompt, because 5-shot
  prompts put the question after the examples and MiniLM reads 256 tokens from the
  start. Production EmbeddingTriage embeds from the start.
* The learned checker's marginal cost is counted as $0 (local CPU inference).
* Leakage: the shipped triage heads were trained with MMLU rows; RouterBench has MMLU.

Security: RouterBench ships only as a pickle, and unpickling runs code. The loader
unpickles with an allowlist: any global outside pandas/numpy/builtins aborts the load.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

TIERS = ("cheap", "middle", "frontier")
DEFAULT_LADDER = {
    "cheap": "mistralai/mistral-7b-chat",
    "middle": "mistralai/mixtral-8x7b-chat",
    "frontier": "gpt-4-1106-preview",
}
ALL_MODELS = (
    "WizardLM/WizardLM-13B-V1.2", "claude-instant-v1", "claude-v1", "claude-v2",
    "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat", "mistralai/mistral-7b-chat", "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
)
# Derived from the Jev tutorial's sample response (476 input tokens -> $0.000019992),
# NOT from the live model page. Pass --jev-price-per-mtok to use the current price.
JEV_PRICE_PER_MTOK = 0.042
LAMBDAS = (None, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)  # None = catalog value
THRESHOLDS = tuple(round(x, 2) for x in np.arange(0.05, 1.0, 0.05))
ENCODER = "all-MiniLM-L6-v2"
_WRONG_BELOW = 0.5  # RouterBench scores are 0..1; some evals grade fractionally
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
    """RouterBench prompts are str(list_of_turns)."""
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
    graded = list(ALL_MODELS) + [f"{m}|total_cost" for m in ALL_MODELS]
    missing = df[graded].isna().any(axis=1)
    if missing.any():
        # 5-shot has 28 arc-challenge rows with no score/cost for any model.
        print(f"dropping {int(missing.sum())} rows with missing scores or costs", flush=True)
        df = df[~missing]
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    df = df.reset_index(drop=True)
    df["text"] = df["prompt"].map(_user_text)
    return df


def _group(eval_name: str) -> str:
    for prefix, g in (("mmlu", "mmlu"), ("chinese", "chinese"), ("Chinese", "chinese"), ("mtbench", "mtbench")):
        if eval_name.startswith(prefix):
            return g
    return eval_name


# ---------------------------------------------------------------------------
# Embeddings — computed once, cached on disk
# ---------------------------------------------------------------------------

_encoder = None


def embed(texts: list[str], cache_dir: Path, tag: str) -> np.ndarray:
    global _encoder
    key = hashlib.sha1(("\x00".join(texts) + ENCODER).encode()).hexdigest()[:16]
    path = cache_dir / f"{tag}-{key}.npy"
    if path.exists():
        return np.load(path)
    if _encoder is None:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(ENCODER)
    x = _encoder.encode(texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x


class PrecomputedTriage:
    """EmbeddingTriage's heads applied to precomputed embeddings."""

    def __init__(self, texts: list[str], x: np.ndarray, kind: str):
        from smartroute.decision import EmbeddingTriage, HeuristicTriage, TriageSignals

        if kind == "heuristic":
            h = HeuristicTriage()
            self._signals = {t: h.signals(t) for t in texts}
            self.calibration, self._backend = {}, "heuristic"
            return
        inner = EmbeddingTriage()
        if inner._heads is None:
            raise SystemExit("no trained triage model; run harness/train_triage.py or pass --triage heuristic")
        self.calibration = inner.calibration
        pl = inner._heads["lookup"].predict_proba(x)[:, 1]
        pd_ = inner._heads["decompose"].predict_proba(x)[:, 1]
        self._signals = {
            t: TriageSignals(p_lookup=float(a), p_decompose=float(b), backend="embedding")
            for t, a, b in zip(texts, pl, pd_)
        }
        self._backend = "embedding"

    def signals(self, prompt: str):
        return self._signals[prompt]

    def backend(self) -> str:
        return self._backend


# ---------------------------------------------------------------------------
# Learned checker
# ---------------------------------------------------------------------------

def fit_checker(features: np.ndarray, correct: np.ndarray, seed: int):
    """Pick the better of logistic regression and a small MLP by validation AUC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    val = rng.random(len(correct)) < 0.15
    candidates = {
        "logreg": LogisticRegression(C=1.0, max_iter=2000),
        "mlp": MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True,
                             max_iter=200, random_state=seed),
    }
    scores = {}
    for name, clf in candidates.items():
        clf.fit(features[~val], correct[~val])
        scores[name] = float(roc_auc_score(correct[val], clf.predict_proba(features[val])[:, 1]))
    best = max(scores, key=scores.get)
    head = candidates[best].fit(features, correct)  # refit on all training rows
    return head, best, scores


# ---------------------------------------------------------------------------
# Curves and AIQ
# ---------------------------------------------------------------------------

def nd_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper convex hull (monotone chain), the frontier reachable by mixing points."""
    pts = sorted(set(points))
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (ox, oy), (ax, ay) = hull[-2], hull[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def curve(points, grid: np.ndarray) -> np.ndarray:
    hull = nd_hull(points)
    xs, ys = zip(*hull)
    vals = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return np.maximum.accumulate(vals)  # non-decreasing: spending more never hurts


def aiq(points, lo: float, hi: float) -> tuple[float, np.ndarray, np.ndarray]:
    grid = np.linspace(lo, hi, 2001)
    vals = curve(points, grid)
    return float(np.trapezoid(vals, grid) / (hi - lo)), grid, vals


def cost_to_reach(grid, vals, target) -> float | None:
    hit = np.nonzero(vals >= target - 1e-12)[0]
    return float(grid[hit[0]]) if len(hit) else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(df, ladder, args) -> dict:
    from sklearn.metrics import roc_auc_score

    from smartroute.decision import DecisionLayer
    from smartroute.escalation import pair_features, tail

    cache = args.cache_dir
    texts = df["text"].tolist()
    emb_texts = [tail(t, args.tail_chars) for t in texts]
    resp_col = f"{ladder['middle']}|model_response"
    responses = [tail(str(r) if r is not None else "", args.tail_chars) for r in df[resp_col].tolist()]

    print("embedding prompts ...", flush=True)
    P = embed(emb_texts, cache, "prompt")
    print(f"embedding {ladder['middle']} answers ...", flush=True)
    R = embed(responses, cache, "resp-" + ladder["middle"].replace("/", "_"))

    rng = np.random.default_rng(args.seed)
    test = rng.random(len(df)) < 0.3
    train = ~test

    S = np.stack([df[ladder[t]].to_numpy(float) for t in TIERS], axis=1)          # (n, 3)
    C = np.stack([df[f"{ladder[t]}|total_cost"].to_numpy(float) for t in TIERS], axis=1)
    mid_wrong = S[:, 1] < _WRONG_BELOW
    jev_cost = (df["text"].str.len().to_numpy() + np.array([len(r) for r in responses])) / 4.0
    jev_cost = jev_cost * args.jev_price_per_mtok / 1_000_000

    # -- checker: train on 70%, score the 30% ------------------------------
    print("training checker ...", flush=True)
    feats = pair_features(P, R)
    head, head_kind, val_auc = fit_checker(feats[train], ~mid_wrong[train], args.seed)
    p_correct = head.predict_proba(feats[test])[:, 1]
    test_auc = float(roc_auc_score(~mid_wrong[test], p_correct))
    if args.save_checker:
        import joblib

        joblib.dump({"head": head, "encoder": ENCODER, "tail_chars": args.tail_chars,
                     "trained_on": f"routerbench:{args.data.name}:{ladder['middle']}:train70"},
                    args.save_checker)

    # -- test-split arrays ---------------------------------------------------
    St, Ct, Pt = S[test], C[test], np.array(texts, dtype=object)[test]
    wrong_t, jev_t = mid_wrong[test], jev_cost[test]
    groups_t = df.loc[test, "eval_name"].map(_group).to_numpy()
    first: dict[str, int] = {}
    for i, t in enumerate(texts):
        first.setdefault(t, i)
    triage = PrecomputedTriage(list(first), P[list(first.values())], args.triage)

    def tiers_for(lmbda) -> tuple[np.ndarray, collections.Counter, DecisionLayer]:
        layer = DecisionLayer(triage=triage, lambda_wrong_usd=lmbda)
        paths = [layer.decide(t).path for t in Pt]
        idx = np.array([TIERS.index("frontier" if p == "decompose" else p) for p in paths])
        return idx, collections.Counter(paths), layer

    def point(idx, escalate=None, extra_cost=None):
        s = St[np.arange(len(idx)), idx].copy()
        c = Ct[np.arange(len(idx)), idx].copy()
        if escalate is not None:
            m = (idx == 1) & escalate
            s[m] = St[m, 2]
            c[idx == 1] += 0.0 if extra_cost is None else extra_cost[idx == 1]
            c[m] += Ct[m, 2]
        return float(s.mean()), float(c.mean()), s

    fixed = {t: (float(St[:, i].mean()), float(Ct[:, i].mean())) for i, t in enumerate(TIERS)}
    lo, hi = fixed["cheap"][1], fixed["frontier"][1]
    endpoints = [(fixed["cheap"][1], fixed["cheap"][0]), (fixed["frontier"][1], fixed["frontier"][0])]
    all11 = [(float(df.loc[test, f"{m}|total_cost"].mean()), float(df.loc[test, m].mean())) for m in ALL_MODELS]

    curves: dict[str, list[tuple[float, float]]] = {
        "zero (ladder)": [(c, a) for a, c in fixed.values()],
        "zero (all 11)": all11,
        "decision": list(endpoints),
        "decision+learned": list(endpoints),
        "decision+perfect": list(endpoints),
    }
    sweep = []
    catalog = None
    for lmbda in LAMBDAS:
        idx, counts, layer = tiers_for(lmbda)
        acc, cost, s_route = point(idx)
        curves["decision"].append((cost, acc))
        pa, pc, _ = point(idx, escalate=wrong_t)
        curves["decision+perfect"].append((pc, pa))
        row = {"lambda": layer._lambda, "paths": dict(counts), "accuracy": acc, "cost": cost,
               "perfect": {"accuracy": pa, "cost": pc}, "learned": {}}
        for th in THRESHOLDS:
            la, lc, _ = point(idx, escalate=p_correct < th)
            curves["decision+learned"].append((lc, la))
            row["learned"][f"{th:.2f}"] = {"accuracy": la, "cost": lc,
                                          "escalation_rate": float((p_correct[idx == 1] < th).mean())}
        sweep.append(row)
        if lmbda is None:
            catalog = (idx, counts, layer, s_route)

    aiqs, reach = {}, {}
    target = 0.95 * fixed["frontier"][0]
    for name, pts in curves.items():
        v, grid, vals = aiq(pts, lo, hi)
        aiqs[name] = v
        reach[name] = cost_to_reach(grid, vals, target)

    # -- headline arms at the catalog lambda --------------------------------
    idx, counts, layer, s_route = catalog
    arms = {f"always-{t}": {"accuracy": a, "cost": c} for t, (a, c) in fixed.items()}
    arms["routed"] = {"accuracy": sweep[0]["accuracy"], "cost": sweep[0]["cost"]}
    # Headline checker threshold: most accurate setting that costs at most half of
    # always-frontier. Chosen on the test split, so it flatters slightly; the AIQ and
    # the full sweep are the threshold-free numbers.
    learned = sweep[0]["learned"]
    affordable = [k for k, v in learned.items() if v["cost"] <= 0.5 * fixed["frontier"][1]]
    best_th = (max(affordable, key=lambda k: learned[k]["accuracy"]) if affordable
               else min(learned, key=lambda k: learned[k]["cost"]))
    arms[f"routed+learned@{best_th} (<=1/2 frontier $)"] = learned[best_th]
    arms["routed+perfect checker"] = sweep[0]["perfect"]
    oracle_idx = np.where(St[:, 0] > 0, 0, np.where(St[:, 1] > 0, 1, np.where(St[:, 2] > 0, 2, 0)))
    oa, oc, _ = point(oracle_idx)
    arms["oracle (ladder)"] = {"accuracy": oa, "cost": oc}

    by_group = {}
    for g in sorted(set(groups_t), key=lambda g: -(groups_t == g).sum()):
        m = groups_t == g
        by_group[g] = {"n": int(m.sum()), "cheap": float(St[m, 0].mean()), "middle": float(St[m, 1].mean()),
                       "frontier": float(St[m, 2].mean()), "routed": float(s_route[m].mean())}

    # -- simulated checker grid (Jev-priced) --------------------------------
    sim = {}
    for detect in (1.0, 0.9, 0.8, 0.7, 0.5):
        for fa in (0.0, 0.1, 0.2, 0.3):
            esc_p = np.where(wrong_t, detect, fa)
            s = St[np.arange(len(idx)), idx].copy()
            c = Ct[np.arange(len(idx)), idx].copy()
            mm = idx == 1
            s[mm] = (1 - esc_p[mm]) * s[mm] + esc_p[mm] * St[mm, 2]
            c[mm] += jev_t[mm] + esc_p[mm] * Ct[mm, 2]
            sim[f"{detect:.1f}/{fa:.1f}"] = {"accuracy": float(s.mean()), "cost": float(c.mean())}

    live = live_escalation(df, ladder, np.nonzero(test)[0][idx == 1], args) if args.jev_verify else None

    return {
        "data": args.data.name, "n_total": len(df), "n_test": int(test.sum()), "ladder": ladder,
        "triage": triage.backend(), "tail_chars": args.tail_chars,
        "lookup_threshold": layer._lookup_threshold, "path_counts": dict(counts),
        "checker": {"kind": head_kind, "val_auc": val_auc, "test_auc": test_auc,
                    "middle_wrong_rate_test": float(wrong_t.mean())},
        "arms": arms, "by_group": by_group, "aiq": aiqs, "cost_range": [lo, hi],
        "cost_to_95pct_frontier": reach, "frontier_accuracy": fixed["frontier"][0],
        "lambda_sweep": sweep, "jev_sim": sim, "jev_live": live,
        "jev_price_per_mtok": args.jev_price_per_mtok,
    }


# ---------------------------------------------------------------------------
# Live Jev checks (costs money; needs OPENROUTER_API_KEY)
# ---------------------------------------------------------------------------

def live_escalation(df, ladder, middle_rows, args) -> dict:
    import asyncio
    import os
    import random

    from smartroute.escalation import JevVerifier

    rows = sorted(random.Random(args.seed).sample(list(middle_rows), min(args.jev_verify, len(middle_rows))))
    verifier = JevVerifier(os.environ.get("OPENROUTER_API_KEY", ""), threshold=0.5, model=args.decision_model)
    col = f"{ladder['middle']}|model_response"

    async def run():
        sem = asyncio.Semaphore(args.concurrency)

        async def one(i):
            async with sem:
                r = await verifier.check(df.at[i, "text"], str(df.at[i, col] or ""))
                return i, r.p_correct, r.cost_usd
        return await asyncio.gather(*(one(i) for i in rows))

    ok = [(i, p, c) for i, p, c in asyncio.run(run()) if p is not None]
    if not ok:
        raise SystemExit("every Jev check failed — is OPENROUTER_API_KEY set?")
    ids = [i for i, _, _ in ok]
    p = np.array([q for _, q, _ in ok])
    cost = np.array([c for _, _, c in ok])
    mid, front = df.loc[ids, ladder["middle"]].to_numpy(float), df.loc[ids, ladder["frontier"]].to_numpy(float)
    cm = df.loc[ids, f"{ladder['middle']}|total_cost"].to_numpy(float)
    cf = df.loc[ids, f"{ladder['frontier']}|total_cost"].to_numpy(float)
    wrong = mid < _WRONG_BELOW
    from sklearn.metrics import roc_auc_score

    sweep = {}
    for th in (0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        e = p < th
        sweep[f"{th:.2f}"] = {
            "accuracy": float(np.where(e, front, mid).mean()),
            "cost": float((cm + cost + e * cf).mean()),
            "escalation_rate": float(e.mean()),
            "detection_rate": float(e[wrong].mean()) if wrong.any() else 0.0,
            "false_alarm_rate": float(e[~wrong].mean()) if (~wrong).any() else 0.0,
        }
    return {"n_checked": len(ok), "auc": float(roc_auc_score(~wrong, p)) if 0 < wrong.sum() < len(wrong) else None,
            "jev_cost_per_check": float(cost.mean()), "threshold_sweep": sweep}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render(r: dict) -> None:
    print(f"\nRouterBench {r['data']} — test n={r['n_test']} of {r['n_total']}, triage={r['triage']}, "
          f"embed tail={r['tail_chars']} chars")
    print("ladder: " + " -> ".join(f"{k}={v}" for k, v in r["ladder"].items()))
    print("paths @ catalog lambda: " + ", ".join(f"{k}={v}" for k, v in sorted(r["path_counts"].items())))
    ck = r["checker"]
    print(f"learned checker: {ck['kind']}, test AUC {ck['test_auc']:.3f} "
          f"(middle wrong on {ck['middle_wrong_rate_test']:.1%} of test prompts)\n")

    base = r["arms"]["always-cheap"]["cost"]
    print(f"{'arm':<40}{'accuracy':>10}{'$/query':>12}{'vs cheap':>10}")
    for name, a in r["arms"].items():
        print(f"{name:<40}{a['accuracy']:>10.1%}{a['cost']:>12.6f}{a['cost'] / base:>9.2f}x")

    lo, hi = r["cost_range"]
    print(f"\nAIQ over ${lo:.6f}..${hi:.6f}/query (higher is better); "
          f"cost to reach 95% of frontier accuracy ({0.95 * r['frontier_accuracy']:.1%})")
    zero = r["aiq"]["zero (ladder)"]
    for name, v in r["aiq"].items():
        c = r["cost_to_95pct_frontier"][name]
        reach = f"${c:.6f} ({c / hi:.0%} of frontier)" if c is not None else "never"
        print(f"  {name:<20} AIQ {v:.4f}  ({v - zero:+.4f} vs zero-ladder)   95%: {reach}")

    print(f"\n{'group':<22}{'n':>6}{'cheap':>8}{'middle':>8}{'front':>8}{'routed':>8}")
    for g, v in r["by_group"].items():
        print(f"{g:<22}{v['n']:>6}{v['cheap']:>8.1%}{v['middle']:>8.1%}{v['frontier']:>8.1%}{v['routed']:>8.1%}")

    if r.get("jev_live"):
        live = r["jev_live"]
        print(f"\nLive Jev — n={live['n_checked']}, AUC {live['auc']}, ${live['jev_cost_per_check']:.6f}/check")
        for th, v in live["threshold_sweep"].items():
            print(f"  th={th} acc {v['accuracy']:.1%} ${v['cost']:.6f} escalate {v['escalation_rate']:.1%} "
                  f"detect {v['detection_rate']:.1%} false+ {v['false_alarm_rate']:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="routerbench_5shot.pkl (or _0shot)")
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--triage", choices=("embedding", "heuristic"), default="embedding")
    ap.add_argument("--cheap", default=DEFAULT_LADDER["cheap"])
    ap.add_argument("--middle", default=DEFAULT_LADDER["middle"])
    ap.add_argument("--frontier", default=DEFAULT_LADDER["frontier"])
    ap.add_argument("--tail-chars", type=int, default=1000, help="embed only the prompt's last N chars (0 = all)")
    ap.add_argument("--jev-price-per-mtok", type=float, default=JEV_PRICE_PER_MTOK)
    ap.add_argument("--jev-verify", type=int, default=0, metavar="N",
                    help="run N live Jev checks on test-split middle answers (costs money; OPENROUTER_API_KEY)")
    ap.add_argument("--decision-model", default="~typesafe/jev-latest")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--save-checker", type=Path, default=None, help="write the fitted checker (joblib)")
    ap.add_argument("--cache-dir", type=Path, default=_REPO_ROOT / "harness" / "results" / ".cache")
    ap.add_argument("--out", type=Path, default=_REPO_ROOT / "harness" / "results" / "routerbench.json")
    args = ap.parse_args()

    ladder = {"cheap": args.cheap, "middle": args.middle, "frontier": args.frontier}
    df = load(args.data, args.n, args.seed)
    result = evaluate(df, ladder, args)
    render(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
