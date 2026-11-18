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
Prompts are split 70/30 at random, --splits times (default 5) with different seeds. The
learned checker retrains on each 70%; every number is on that split's held-out 30%.
Intervals are 95% CIs across splits with the Nadeau-Bengio correction for overlapping
test sets; "vs" columns are paired differences within each split.

--jev scores every middle answer once with Jev (TypeSafe's API when TYPESAFE_API_KEY is
set, else OpenRouter), caches the scores, and reuses them across splits — Jev is not
trained, so it needs no split of its own. Costs ~$0.00005 per row.

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
to always-cheap/always-frontier, and every checker can be switched off, so
AIQ(decision+checker) >= AIQ(decision) by construction: equal means the checker adds
nothing. The decision router itself can score below zero (ladder) if its routed points
sit under the ladder hull.

Consistency checkers compare the middle answer with other models' *recorded* answers
(RouterBench has one answer per model, so same-model resampling cannot be replayed),
and pay those models' recorded cost on every middle-routed prompt.

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
JEV_PRICE_PER_MTOK = 0.04198  # Jev 1.14 OpenRouter list price, input only (2026-11-18)
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

# ---------------------------------------------------------------------------
# Jev: scored once per row, cached, reused by every split
# ---------------------------------------------------------------------------

def jev_scores(df, ladder, rows, args) -> tuple[np.ndarray, np.ndarray, int]:
    """
    P(correct) and billed cost for the middle answer on `rows`. Jev is not trained, so
    one score per (row, middle model) serves every split; results are cached on disk.
    Failed checks get p=1.0 (no escalation), matching the production failure policy.
    """
    import asyncio
    import os

    from smartroute.escalation import JevVerifier

    key = os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    provider = "typesafe" if os.environ.get("TYPESAFE_API_KEY") else "openrouter"
    col = f"{ladder['middle']}|model_response"
    cache = args.cache_dir / f"jev-{provider}-{ladder['middle'].replace('/', '_')}.json"
    done: dict[str, list] = json.loads(cache.read_text()) if cache.exists() else {}
    ids = [str(df.at[i, "sample_id"]) for i in rows]
    todo = [(i, sid) for i, sid in zip(rows, ids) if sid not in done]

    if todo:
        verifier = JevVerifier(key, threshold=0.5, model=args.decision_model, provider=provider,
                               price_per_mtok=args.jev_price_per_mtok)
        print(f"Jev via {provider}: {len(todo)} checks ({len(rows) - len(todo)} cached) ...", flush=True)

        async def run():
            sem = asyncio.Semaphore(args.concurrency)

            async def one(i, sid):
                async with sem:
                    for attempt in range(4):
                        r = await verifier.check(df.at[i, "text"], _user_text(str(df.at[i, col] or "")))
                        if r.p_correct is not None:
                            return sid, [r.p_correct, r.cost_usd]
                        await asyncio.sleep(2 ** attempt)
                    return sid, None
            out = []
            batch = 500
            for start in range(0, len(todo), batch):
                out += await asyncio.gather(*(one(i, sid) for i, sid in todo[start:start + batch]))
                done.update({sid: v for sid, v in out[-batch:] if v is not None})
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(done))  # checkpoint: a crash keeps what was paid for
                print(f"  {min(start + batch, len(todo))}/{len(todo)}", flush=True)
            return out
        asyncio.run(run())

    p = np.array([done[sid][0] if sid in done else 1.0 for sid in ids])
    cost = np.array([done[sid][1] if sid in done else 0.0 for sid in ids])
    return p, cost, sum(sid not in done for sid in ids)


# ---------------------------------------------------------------------------
# Confidence intervals across random splits
# ---------------------------------------------------------------------------

def corrected_ci(values, n_train: int, n_test: int) -> dict:
    """
    95% CI from repeated random splits with the Nadeau-Bengio correction: test sets
    overlap between splits, so the naive variance understates the uncertainty. The
    corrected variance multiplies by (1/k + n_test/n_train) instead of 1/k.
    """
    from scipy import stats

    v = np.asarray(values, dtype=float)
    k = len(v)
    mean = float(v.mean())
    if k < 2:
        return {"mean": mean, "lo": None, "hi": None, "k": k}
    se = float(np.sqrt((1 / k + n_test / n_train) * v.var(ddof=1)))
    t = float(stats.t.ppf(0.975, k - 1))
    return {"mean": mean, "lo": mean - t * se, "hi": mean + t * se, "k": k}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def default_peers(df, ladder, k: int = 2) -> list[str]:
    """The k non-ladder models whose mean cost is closest to the middle model's."""
    mid_cost = df[f"{ladder['middle']}|total_cost"].mean()
    others = [m for m in ALL_MODELS if m not in ladder.values()]
    return sorted(others, key=lambda m: abs(df[f"{m}|total_cost"].mean() - mid_cost))[:k]


def evaluate(df, ladder, args) -> dict:
    from sklearn.metrics import roc_auc_score

    from smartroute.decision import DecisionLayer
    from smartroute.escalation import answers_agree, pair_features, tail

    n = len(df)
    texts = df["text"].tolist()
    resp_col = f"{ladder['middle']}|model_response"

    # -- per-row quantities, computed once ------------------------------------
    print("embedding prompts ...", flush=True)
    P = embed([tail(t, args.tail_chars) for t in texts], args.cache_dir, "prompt")
    print(f"embedding {ladder['middle']} answers ...", flush=True)
    R = embed([tail(str(r) if r is not None else "", args.tail_chars) for r in df[resp_col]],
              args.cache_dir, "resp-" + ladder["middle"].replace("/", "_"))
    feats = pair_features(P, R)

    S = np.stack([df[ladder[t]].to_numpy(float) for t in TIERS], axis=1)
    C = np.stack([df[f"{ladder[t]}|total_cost"].to_numpy(float) for t in TIERS], axis=1)
    wrong = S[:, 1] < _WRONG_BELOW
    groups = df["eval_name"].map(_group).to_numpy()

    # Consistency: agreement with other models' *recorded* answers (one answer per
    # model, so same-model resampling cannot be replayed).
    mid_answers = [_user_text(str(r)) for r in df[resp_col]]
    peers = args.sc_peers or default_peers(df, ladder)

    def agreement(models):
        votes, cost = np.zeros(n), np.zeros(n)
        for m in models:
            other = df[f"{m}|model_response"]
            votes += [answers_agree(a, _user_text(str(b))) for a, b in zip(mid_answers, other)]
            cost += df[f"{m}|total_cost"].to_numpy(float)
        return votes / len(models), cost

    print(f"scoring agreement with {peers} and {ladder['cheap']} ...", flush=True)
    p_peers, cost_peers = agreement(peers)
    p_cheap, cost_cheap = agreement([ladder["cheap"]])

    first: dict[str, int] = {}
    for i, t in enumerate(texts):
        first.setdefault(t, i)
    triage = PrecomputedTriage(list(first), P[list(first.values())], args.triage)
    routes = {}
    for lmbda in LAMBDAS:
        layer = DecisionLayer(triage=triage, lambda_wrong_usd=lmbda)
        paths = [layer.decide(t).path for t in texts]
        routes[lmbda] = (np.array([TIERS.index("frontier" if q == "decompose" else q) for q in paths]),
                         paths, layer)

    jev = None
    if args.jev:
        mid_rows = np.arange(n)  # every row: which rows are test, and which go middle, varies
        if args.jev_limit:
            mid_rows = np.random.default_rng(0).choice(mid_rows, size=min(args.jev_limit, n), replace=False)
        p_j, c_j, failed = jev_scores(df, ladder, sorted(mid_rows.tolist()), args)
        jev = {"rows": np.array(sorted(mid_rows.tolist())), "p": p_j, "cost": c_j, "failed": failed}

    # -- one split ----------------------------------------------------------
    def run_split(split_seed: int) -> dict:
        rng = np.random.default_rng(split_seed)
        test = rng.random(n) < 0.3
        if jev is not None:  # with a Jev subsample, evaluate only where Jev scored
            scored = np.zeros(n, dtype=bool)
            scored[jev["rows"]] = True
            test &= scored
        train = ~test
        rows = np.nonzero(test)[0]

        head, head_kind, _ = fit_checker(feats[train], ~wrong[train], split_seed)
        p_learned = head.predict_proba(feats[rows])[:, 1]
        checkers = {
            "learned": (p_learned, np.zeros(len(rows))),
            "consistency[cheap]": (p_cheap[rows], cost_cheap[rows]),
            "consistency[peers]": (p_peers[rows], cost_peers[rows]),
            "learned+consistency[peers]": ((p_learned + p_peers[rows]) / 2, cost_peers[rows]),
        }
        if jev is not None:
            pos = {r: k for k, r in enumerate(jev["rows"])}
            sel = [pos[r] for r in rows]
            pj, cj = jev["p"][sel], jev["cost"][sel]
            checkers["jev"] = (pj, cj)
            checkers["jev+learned"] = ((pj + p_learned) / 2, cj)
            checkers["jev+consistency[cheap]"] = ((pj + p_cheap[rows]) / 2, cj + cost_cheap[rows])

        St, Ct, wt = S[rows], C[rows], wrong[rows]

        def point(idx, escalate=None, extra=None):
            s_ = St[np.arange(len(idx)), idx].copy()
            c_ = Ct[np.arange(len(idx)), idx].copy()
            if escalate is not None:
                m = (idx == 1) & escalate
                s_[m] = St[m, 2]
                if extra is not None:
                    c_[idx == 1] += extra[idx == 1]
                c_[m] += Ct[m, 2]
            return float(s_.mean()), float(c_.mean()), s_

        fixed = {t: (float(St[:, i].mean()), float(Ct[:, i].mean())) for i, t in enumerate(TIERS)}
        lo, hi = fixed["cheap"][1], fixed["frontier"][1]
        ends = [(fixed["cheap"][1], fixed["cheap"][0]), (fixed["frontier"][1], fixed["frontier"][0])]
        curves = {
            "zero (ladder)": [(c, a) for a, c in fixed.values()],
            "zero (all 11)": [(float(df[f"{m}|total_cost"].to_numpy(float)[rows].mean()),
                               float(df[m].to_numpy(float)[rows].mean())) for m in ALL_MODELS],
            "decision": list(ends),
            **{f"decision+{k}": list(ends) for k in checkers},
            "decision+perfect": list(ends),
        }
        table = {}
        for lmbda in LAMBDAS:
            idx = routes[lmbda][0][rows]
            acc, cost, s_route = point(idx)
            for name in curves:
                if name.startswith("decision"):
                    curves[name].append((cost, acc))  # every checker can be switched off
            pa, pc, _ = point(idx, escalate=wt)
            curves["decision+perfect"].append((pc, pa))
            for name, (p, extra) in checkers.items():
                for th in THRESHOLDS:
                    ca, cc, _ = point(idx, escalate=p < th, extra=extra)
                    curves[f"decision+{name}"].append((cc, ca))
                    if lmbda is None:
                        table.setdefault(name, {})[f"{th:.2f}"] = {"accuracy": ca, "cost": cc}
            if lmbda is None:
                catalog_idx, catalog_route = idx, (acc, cost, s_route, pa, pc)

        target = 0.95 * fixed["frontier"][0]
        aiqs, reach = {}, {}
        for name, pts in curves.items():
            v, grid, vals = aiq(pts, lo, hi)
            c95 = cost_to_reach(grid, vals, target)
            aiqs[name], reach[name] = v, (c95 / hi if c95 is not None else None)

        acc, cost, s_route, pa, pc = catalog_route
        arms = {f"always-{t}": {"accuracy": a, "cost": c} for t, (a, c) in fixed.items()}
        arms["routed"] = {"accuracy": acc, "cost": cost}
        for name, tab in table.items():
            ok = [k for k, v in tab.items() if v["cost"] <= 0.5 * hi]
            best = max(ok, key=lambda k: tab[k]["accuracy"]) if ok else min(tab, key=lambda k: tab[k]["cost"])
            arms[f"routed+{name}@{best}"] = tab[best]
        arms["routed+perfect checker"] = {"accuracy": pa, "cost": pc}
        oracle = np.where(St[:, 0] > 0, 0, np.where(St[:, 1] > 0, 1, np.where(St[:, 2] > 0, 2, 0)))
        oa, oc, _ = point(oracle)
        arms["oracle (ladder)"] = {"accuracy": oa, "cost": oc}

        by_group = {}
        for g in sorted(set(groups[rows]), key=lambda g: -(groups[rows] == g).sum()):
            m = groups[rows] == g
            by_group[g] = {"n": int(m.sum()), "cheap": float(St[m, 0].mean()), "middle": float(St[m, 1].mean()),
                           "frontier": float(St[m, 2].mean()), "routed": float(s_route[m].mean())}
        return {
            "seed": split_seed, "n_test": int(len(rows)), "n_train": int(train.sum()),
            "learned_kind": head_kind, "middle_wrong_rate": float(wt.mean()),
            "auc": {k: float(roc_auc_score(~wt, p)) for k, (p, _) in checkers.items()},
            "aiq": aiqs, "cost_to_95pct": reach, "arms": arms, "by_group": by_group,
            "frontier_accuracy": fixed["frontier"][0], "cost_range": [lo, hi],
            "paths": dict(collections.Counter(np.array(routes[None][1], dtype=object)[rows])),
        }

    splits = []
    for k in range(args.splits):
        print(f"split {k + 1}/{args.splits} ...", flush=True)
        splits.append(run_split(args.seed + k))

    # -- aggregate ------------------------------------------------------------
    n_tr = int(np.mean([sp["n_train"] for sp in splits]))
    n_te = int(np.mean([sp["n_test"] for sp in splits]))
    ci = lambda vals: corrected_ci(vals, n_tr, n_te)  # noqa: E731
    names = list(splits[0]["aiq"])
    summary = {
        "aiq": {k: ci([sp["aiq"][k] for sp in splits]) for k in names},
        # paired: same split, so split-to-split variation cancels
        "aiq_vs_zero_all11": {k: ci([sp["aiq"][k] - sp["aiq"]["zero (all 11)"] for sp in splits]) for k in names},
        "aiq_vs_decision": {k: ci([sp["aiq"][k] - sp["aiq"]["decision"] for sp in splits]) for k in names},
        "cost_to_95pct": {k: ci([sp["cost_to_95pct"][k] if sp["cost_to_95pct"][k] is not None else np.nan
                                 for sp in splits]) for k in names},
        "auc": {k: ci([sp["auc"][k] for sp in splits]) for k in splits[0]["auc"]},
    }
    return {
        "data": args.data.name, "n_total": n, "ladder": ladder, "triage": triage.backend(),
        "tail_chars": args.tail_chars, "sc_peers": peers, "splits": len(splits),
        "jev": None if jev is None else {"scored": int(len(jev["rows"])), "failed": jev["failed"],
                                         "cost_per_check": float(jev["cost"].mean()),
                                         "price_per_mtok": args.jev_price_per_mtok},
        "summary": summary, "per_split": splits,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(c: dict, pct: bool = False, signed: bool = False) -> str:
    f = (lambda x: f"{x:+.1%}" if signed else f"{x:.1%}") if pct else \
        (lambda x: f"{x:+.4f}" if signed else f"{x:.4f}")
    if c["mean"] is None or np.isnan(c["mean"]):
        return "never"
    if c["lo"] is None:
        return f(c["mean"])
    return f"{f(c['mean'])} [{f(c['lo'])}, {f(c['hi'])}]"


def render(r: dict) -> None:
    sp0 = r["per_split"][0]
    print(f"\nRouterBench {r['data']} — {r['n_total']} prompts, {r['splits']} random 70/30 splits "
          f"(test n≈{sp0['n_test']}), triage={r['triage']}")
    print("ladder: " + " -> ".join(f"{k}={v}" for k, v in r["ladder"].items()))
    print(f"consistency peers: {', '.join(r['sc_peers'])}")
    if r["jev"]:
        j = r["jev"]
        print(f"jev: {j['scored']} rows scored ({j['failed']} failed -> no escalation), "
              f"${j['cost_per_check']:.6f}/check at ${j['price_per_mtok']}/M")
    print("intervals: mean [95% CI], Nadeau-Bengio corrected across splits\n")

    print("checker AUC")
    for k, v in r["summary"]["auc"].items():
        print(f"  {k:<30} {_fmt(v)}")

    print(f"\n{'router':<40}{'AIQ':<28}{'vs zero (all 11)':<30}{'vs no checker':<30}cost to 95% of frontier")
    s = r["summary"]
    for k in s["aiq"]:
        print(f"{k:<40}{_fmt(s['aiq'][k]):<28}{_fmt(s['aiq_vs_zero_all11'][k], signed=True):<30}"
              f"{_fmt(s['aiq_vs_decision'][k], signed=True):<30}{_fmt(s['cost_to_95pct'][k], pct=True)}")

    print(f"\nsplit 1 arms (catalog lambda):\n{'arm':<46}{'accuracy':>10}{'$/query':>12}")
    for name, a in sp0["arms"].items():
        print(f"{name:<46}{a['accuracy']:>10.1%}{a['cost']:>12.6f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="routerbench_5shot.pkl (or _0shot)")
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=int, default=5, help="random 70/30 splits for confidence intervals")
    ap.add_argument("--triage", choices=("embedding", "heuristic"), default="embedding")
    ap.add_argument("--cheap", default=DEFAULT_LADDER["cheap"])
    ap.add_argument("--middle", default=DEFAULT_LADDER["middle"])
    ap.add_argument("--frontier", default=DEFAULT_LADDER["frontier"])
    ap.add_argument("--tail-chars", type=int, default=1000, help="embed only the prompt's last N chars (0 = all)")
    ap.add_argument("--sc-peers", type=lambda v: [m.strip() for m in v.split(",") if m.strip()],
                    default=None, help="models whose recorded answers the consistency checker compares "
                                       "against (default: 2 non-ladder models priced nearest the middle)")
    ap.add_argument("--jev", action="store_true",
                    help="score middle answers with Jev (costs money: TYPESAFE_API_KEY or OPENROUTER_API_KEY)")
    ap.add_argument("--jev-limit", type=int, default=0, help="score only this many random rows with Jev")
    ap.add_argument("--jev-price-per-mtok", type=float, default=JEV_PRICE_PER_MTOK)
    ap.add_argument("--decision-model", default="~typesafe/jev-latest")
    ap.add_argument("--concurrency", type=int, default=16)
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
