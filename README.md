# SmartRoute

### An inference router that sends every request to the cheapest model that can actually handle it.

**One endpoint. Every provider. The frontier model only when it earns its place.**

SmartRoute triages each request, answers the obvious lookups with the cheapest model,
defaults everything else to a capable mid-tier model, and splits genuinely multi-part
work into sub-tasks that are individually routed — but only when the arithmetic says
splitting is cheaper than just asking a frontier model.

```
prompt
  ├─ LAYER 1  confident "google-substitute" lookup?  ──→  CHEAP
  ├─ GATE     sub-tasks needed AND worth paying for? ──→  MIDDLE   ← the default
  └─ LAYER 2  decompose, route each sub-task         ──→  cheap | middle | frontier
                                                          then synthesise
```

The frontier tier is never a default destination. It is reached only when the gate's
cost arithmetic prefers it, or via layer 2's bail-out.

## Why it is built this way

Every design choice below came out of measuring the previous one, not from a hunch.

**The regex classifier could not classify.** Run over all 500 rows of its own MMLU set,
the original keyword classifier sent **92.8%** of prompts to MEDIUM (the dataset's own
labels say 43%). Reordering the patterns does not help — `EASY` patterns are anchored
`.match()` on openers like "what is", and real prompts are declarative stems, so nothing
matches either way.

**Worse, it could not see a hidden hop.** These two open identically:

```
lookup   "In what year was Harvard founded?"
2-hop    "In what year was the university where Tokarev was a professor founded?"
```

so **41.6%** of HotpotQA's multi-hop questions were being routed to the *cheapest* model.

**Routing accuracy, measured offline over 1500 rows** (`harness/eval_routing.py --compare`):

|                             | regex heuristic | learned heads |
|-----------------------------|-----------------|---------------|
| MMLU → middle               | 86.6%           | **100.0%**    |
| HotpotQA → decompose        | 10.2%           | **53.4%**     |
| overall                     | 47.6%           | 56.4%         |
| **multi-hop sent to CHEAP** | **41.6%**       | **1.0%**      |

TriviaQA recall is deliberately low. Layer 1 only fires when its *measured* precision
clears the bar the catalog economics set — diverting a lookup saves $0.000133, so at
λ=$0.001 the head needs 88.3% precision. Leaving savings on the table is the correct
side of an asymmetric loss: a missed lookup costs a fraction of a cent, a false one
returns a bad answer.

## Thresholds are derived, not guessed

Breaking even on a lookup divert requires `precision = λ / (saving + λ)`:

| λ (your cost of a wrong answer) | required precision |
|---|---|
| $0.0005 | 79.0% |
| $0.0010 | 88.3% |
| $0.0100 | 98.7% |

The trainer persists a measured precision/recall curve; the decision layer picks the
lowest threshold that clears the bar — and **disables layer 1 entirely** if no threshold
reaches it. One knob, `lambda_wrong_answer_usd`, in `models.yaml`.

The same logic gates layer 2. Decompose + synthesise costs ~2.4× a direct middle answer
before a single sub-task runs, so splitting only pays when it replaces frontier work
*and* the sub-tasks stay cheap:

| sub-task mix | vs one frontier call | |
|---|---|---|
| 3 × cheap | 0.52× | **pays** |
| 2 × cheap + 1 × middle | 0.59× | **pays** |
| 1 each | 1.21× | loses |

A hard dollar guard runs before any quality weighting: if a plan's projected spend
already exceeds one frontier call, the plan is discarded and that call is made instead.

## Model-agnostic by construction

Every model, price, and tier lives in `models.yaml`. Swapping providers is a config
edit — there is a test that asserts it:

```yaml
models:
  - {alias: cheap,    id: together_ai/openai/gpt-oss-20b,      role: cheap}
  - {alias: middle,   id: together_ai/openai/gpt-oss-120b,     role: middle}
  - {alias: frontier, id: together_ai/Qwen/Qwen3.5-397B-A17B,  role: frontier}
routing:
  tiers: [cheap, middle, frontier]
  default_tier: middle
```

Prices resolve from LiteLLM's registry (4326 models) with a YAML override, and an
unpriced model **raises** rather than silently costing $0.00. The catalog also refuses
to load a judge from the same model family as the tier it grades, and a tier ladder
that does not ascend in price.

## Getting started

```bash
uv sync --extra server --extra ml --extra bench --group dev
cp .env.example .env            # add your provider key

uv run pytest                                          # 207 tests
uv run --extra bench harness/build_corpora.py          # one-time: fetch eval corpora
uv run --extra ml    harness/train_triage.py           # fit the triage heads
uv run --extra ml    harness/eval_routing.py --compare # measure routing, offline, $0
uv run smartroute-server                               # OpenAI-compatible API
```

Routing metadata comes back in the `X-SmartRoute-Meta` response header — path taken,
why, projected cost, sub-task count — so the response body stays 100% OpenAI-compatible.

## Status

The routing layer is built and measured. Still open: the cascade verifier is disabled
by default pending rework — the previous judge (Qwen2.5-72B at $1.20/M) cost **1.44×
the frontier call it existed to avoid**, at every output length, so it could never pay
for itself. End-to-end cost numbers await a live run.

Reach out at kartikay3@outlook.com if you want early access.
