# The SmartRoute benchmark

How to measure whether routing saves you money, what the harness measures, and what it
found.

- [Why this exists](#why-this-exists) — what a routing benchmark has to prove
- [Running it](#running-it) — the three commands
- [The corpus](#the-corpus) — 200 questions, and why not MMLU
- [The arms](#the-arms) — what each one isolates
- [Metrics](#metrics) — what every column means
- [Results](#results) — what was measured, 2026-09-21
- [Adding your own corpus](#adding-your-own-corpus)
- [Methodology notes](#methodology-notes) — mistakes this harness has already made

---

## Why this exists

A router's claim is always the same shape: *"same quality, less money."* Proving it
needs three things, and most published numbers are missing at least one.

**A baseline you can actually lose to.** Beating `always-frontier` is nearly free —
send anything to a cheaper model and you win. The honest question is whether routing
beats *picking one good mid-tier model and never thinking again*. That is the
`always-middle` arm, and it is the one that has repeatedly falsified this project's
assumptions.

**Traffic that resembles your traffic.** A router tuned on academic benchmarks is
tuned on the wrong distribution. See [The corpus](#the-corpus).

**Cost measured, not estimated.** Every arm reports what it actually spent, priced from
the catalog at the call site. Errored prompts keep whatever they spent in the totals —
dropping them makes a failure-prone arm look cheaper, which is how you conclude the
broken thing is the efficient one.

---

## Running it

### Routing accuracy, offline, free

`decide()` makes no network calls, so routing can be evaluated over a whole corpus for
nothing. Start here.

```bash
uv run --extra ml harness/eval_routing.py --compare
```

Reports, per corpus, where each question was routed against where it should have gone,
for both the learned heads and the regex fallback.

### Cost and accuracy, live

```bash
uv run --extra ml harness/run_routing.py --n 200 --arm all --catalog models.anthropic.yaml
```

Prints an estimate and asks before spending. Add `--dry-run` to see the estimate alone,
`--yes` to skip the prompt, `--cost-ceiling N` to raise the refusal threshold.

Results append to `harness/results/routing_<arm>_<timestamp>.jsonl` as each prompt
completes, so a crash costs one result rather than the batch.

### Retraining the triage heads

```bash
uv run --extra ml harness/train_triage.py --corpora synthetic --test-size 0.5
```

Fits both heads, prints a held-out classification report and a measured
precision/recall curve, and writes `src/smartroute/data/triage.joblib`. The curve is
persisted because the decision layer derives its operating threshold from the
economics and needs to know which threshold actually delivers a given precision.

**Retrain whenever the corpus changes.** Heads trained on one distribution do not
transfer to another — [see below](#methodology-notes).

---

## The corpus

200 original questions in `harness/data/synthetic_corpus.py`, materialised to
`harness/data/synthetic_200.jsonl`.

| category | n | expected tier | example |
|---|---|---|---|
| lookup | 50 | cheap | *"What's the capital of Australia?"* |
| explanation | 35 | middle | *"Explain the difference between TCP and UDP."* |
| tech | 35 | middle | *"My Python script says ModuleNotFoundError: No module named requests."* |
| analysis | 25 | middle | *"Should a small startup use microservices or a monolith?"* |
| practical | 15 | middle | *"If I invest $5,000 at 6% compounded yearly, what's it worth in 10 years?"* |
| news | 10 | middle | *"What is the CHIPS Act and what was it meant to achieve?"* |
| multipart | 30 | decompose | *"I'm building a mobile app for 10,000 users. What database, where hosted, and what monthly cost?"* |

### Why not MMLU, TriviaQA or HotpotQA

They were convenient, not representative:

- **MMLU** is 4-way multiple choice. Nobody asks an assistant to pick A/B/C/D.
- **TriviaQA** is deliberately obscure pub trivia. Real lookups are mundane.
- **HotpotQA** welds two facts into one sentence — *"the university where X was a
  professor"* — which reads nothing like what a person types.

This is not academic tidiness. Heads trained on those three caught **7 of 50** lookups
and **0 of 30** multi-part questions when tested on natural phrasing. They had learned
the corpora's surface form, not the underlying distinction, and every routing number
derived from them was measuring the wrong thing.

### Grading

Each question carries `must_include`: a list of requirements, each either a string that
must appear or a list of acceptable alternatives. Matching is case-insensitive over
normalised text (lowercased, punctuation and articles stripped).

```python
("Explain the difference between TCP and UDP.",
 ["tcp", "udp", ["connection", "handshake"], ["reliab", "ordered", "guarantee"]]),
```

Deterministic, free, and no LLM judge — whose cost this project separately disproved.
Assertions are loose enough to accept any correct phrasing and strict enough to reject
a wrong one. Verified adversarially (`tests/test_corpora.py`):

| answer | graded |
|---|---|
| "TCP is connection-oriented and reliable with ordered delivery; UDP is connectionless…" | correct |
| "They are both internet protocols used for sending data." | wrong |
| "The capital of Australia is Sydney." | wrong |
| "It is approximately 400." (for *15% of 2,400*) | wrong |

`expected_path` is the ground-truth routing label. Because it is declared per item, one
corpus both trains the classifier and evaluates it.

---

## The arms

| arm | what it does | what it isolates |
|---|---|---|
| `routed` | the full two-layer decision layer | the thing under test |
| `always-cheap` | every prompt to the cheapest tier | the cost floor, and the accuracy you give up to reach it |
| `always-middle` | every prompt to the middle tier | **the honest baseline** — one good default, no router |
| `always-frontier` | every prompt to the most expensive tier | the quality ceiling, and the bill for it |

Run `--arm all` to get all four against identical prompts in one pass.

---

## Metrics

| column | meaning |
|---|---|
| `accuracy` | share of gradeable prompts whose answer satisfied every assertion |
| `extraction failures` | empty or unusable responses; counted wrong, reported separately |
| `total cost` / `cost per prompt` | actual spend, priced from the catalog at the call site |
| `p50` / `p95 latency` | end-to-end wall clock per request, including routing |
| `paths taken` | which tier each corpus category was routed to |
| `cheap leakage` | work that needed more than the cheapest model but was sent there anyway |

Read the leakage row, not just accuracy. Overall accuracy is dominated by whichever
category is largest; leakage is the number that tells you the router is doing harm.

Per-prompt records in the JSONL carry `p_lookup`, `p_decompose`, `gate_reason`,
`route_path`, `projected_cost_usd` and `subtask_count`, so any decision can be
reconstructed after the fact.

---

## Results

**Run of 2026-09-21** — 200 prompts, Anthropic ladder (`claude-haiku-4-5-20251001` →
`claude-sonnet-5` → `claude-opus-5`), λ=$0.01, temperature 0, max_tokens 2000.
Catalog fingerprint `193bdc5c77ce25b5`.

| arm | $/query | accuracy | vs always-cheap |
|---|---|---|---|
| **always-cheap** | **$0.001206** | **100.0%** | — |
| routed | $0.004949 | 100.0% | 4.10× cost, +0.0pp |
| always-middle | $0.004981 | 100.0% | 4.13× cost, +0.0pp |
| always-frontier | $0.014174 | 100.0% | 11.75× cost, +0.0pp |

**All four arms tied at 100%.** Across 199 completed queries, cheap and middle
disagreed on correctness for **zero**. There is no quality signal to route on, so
routing is 4.1× the cost for nothing.

### What the classifier actually did

| | |
|---|---|
| routing accuracy, held-out 100 | **81.0%** |
| routing accuracy, all 200 | 81.9% |
| lookups sent to cheap | **49 / 50** |
| routed cost on the lookup category | **1.03×** always-cheap |
| multi-part reaching layer 2 | **0 / 30** |

Held-out and full-set accuracy are within a point, so the heads generalise rather than
memorise. Layer 1 captured essentially all available savings on the traffic it was
built for. The failure is not in the classifier.

### Why layer 2 never fired

Only 2 of 199 queries cleared the 0.75 decompose threshold, and both were then
suppressed by the cost guard:

```
decomposable p=0.76 but projected decomposition spend $0.011506
>= one frontier call $0.008140 — splitting cannot pay off
```

Decompose plus synthesis costs roughly 2.4× a direct middle answer before any sub-task
runs. It pays only when it replaces frontier work *and* the sub-tasks stay cheap:

| sub-task mix | vs one frontier call | |
|---|---|---|
| 3 × cheap | 0.52× | pays |
| 2 × cheap + 1 × middle | 0.59× | pays |
| 1 of each | 1.21× | loses |

On Anthropic's 5× cheap→frontier spread that arithmetic never closes. Together AI's
12× spread is wide enough that it might.

### Per-tier accuracy, older run

From the 2026-09-21 academic-corpus run, the only measurement so far where tiers
differed at all:

| corpus | cheap | middle | frontier |
|---|---|---|---|
| triviaqa | 87.9% | 90.9% | 97.0% |
| mmlu | 50.0% | **92.4%** | 89.4% |
| hotpotqa | 28.8% | 43.9% | 54.5% |

On MMLU the middle tier **beats** the frontier — routing those questions up is pure
waste. These are the priors the expected-cost rule should use in place of its current
hand-picked failure multipliers; wiring that loop is unstarted.

### How to read this

The result is conditional, not universal. Routing pays when some traffic fails at the
cheap tier. Three ways that becomes true:

1. **Harder traffic.** Production loads with long context, tricky reasoning, or domain
   depth where a small model measurably fails.
2. **A wider ladder.** Together AI's $0.05 → $0.60 spread leaves far more room than
   Anthropic's $1 → $5.
3. **A quality bar above correctness.** If shallow-but-correct answers cost you
   something, a metric that can see the difference would change the ranking.

Run the harness against your own traffic before believing any of it.

---

## Adding your own corpus

The most useful contribution to this repo is a corpus the cheap tier fails.

**1. Write the questions.** Follow `harness/data/synthetic_corpus.py`: a list of
`(prompt, must_include)` pairs grouped by category, each category mapped to an
`expected_path` of `cheap`, `middle` or `decompose`. Run the module to emit JSONL — it
asserts no duplicate prompts and that every item has assertions, since an item with no
assertions is unconditionally correct and silently inflates accuracy.

**2. Register a dataset class** in `harness/corpora.py` implementing the `Dataset`
protocol:

```python
class MyDataset:
    name = "mine"
    expected_path = ATOMIC          # per-item labels override this

    def load(self, n=None, seed=42) -> list[Example]: ...
    def system_prompt(self) -> str: ...
    def grade(self, example, response) -> tuple[bool, bool]:   # (correct, extraction_failed)
        ...

DATASETS["mine"] = MyDataset
```

**3. Retrain and measure.**

```bash
uv run --extra ml harness/train_triage.py --corpora mine
uv run --extra ml harness/eval_routing.py --compare
uv run --extra ml harness/run_routing.py --n 200 --arm all --datasets mine
```

### Swapping providers

Point `--catalog` at a different YAML. No source changes; `tests/test_catalog.py`
asserts a wholesale swap works, including routing a request end-to-end through it.

```bash
uv run --extra ml harness/run_routing.py --n 200 --arm all --catalog models.groq.yaml
```

---

## Methodology notes

Mistakes this harness has already made, kept here because they are easy to repeat.

**A token cap biases against verbose models.** At `max_tokens=512` the stronger models
hit the cap on 108 of 200 prompts while the cheapest hit it on 8. Truncated answers
miss assertions that appear late, so the better models scored *worse* — `always-frontier`
at 92.6% against `always-cheap` at 99.5%. Set the cap high enough that it never binds
(currently 2000) and check the at-cap count before trusting a run.

**Reasoning tokens can consume the whole budget.** Opus returned 11 empty responses at
the 512 cap, having spent it all on reasoning before emitting text.

**Not every model accepts every parameter.** `claude-sonnet-5` and `claude-opus-5`
reject `temperature` with a hard 400 while `claude-haiku-4-5` accepts it. Pinning
temperature for reproducibility silently destroyed two arms before
`ModelSpec.unsupported_params` existed.

**Arms must use identical settings.** The baselines passed generation params through
`call_model` while the routed arm went through `Router.route_async`, which forwarded
none — so the routed arm ran uncapped. `route_async` now threads params to every tier
call, including layer 2's sub-tasks.

**Classifier heads do not transfer across distributions.** Retrain on the corpus you
intend to measure, and report routing accuracy on a held-out split. The trainer records
`held_out_ids` in the model bundle for exactly this.

**Cost accounting must include failures.** An arm that errors more will look cheaper if
errored prompts are dropped from the totals.

---

*Last reviewed: 2026-09-22. Raw results in `harness/results/`; every figure above is
reproducible from the JSONL there.*
