# The SmartRoute benchmark

How to measure whether routing saves you money, what the harness measures, and what it
found.

- [Why this exists](#why-this-exists) — what a routing benchmark has to prove
- [Running it](#running-it) — the three commands
- [The corpus](#the-corpus) — 200 questions, and why not MMLU
- [The arms](#the-arms) — what each one isolates
- [Metrics](#metrics) — what every column means
- [Results](#results) — what was measured, 2026-09-21
- [RouterBench results](#routerbench-results) — 36k prompts, with and without answer checkers, 2026-11-18
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

Reports, grouped by where each question *should* have gone, where it actually went —
for both the learned heads and the regex fallback, so you can see what the training
bought. Watch the SAFETY block: it counts work that needed more than the cheapest
model and was sent there anyway.

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

Their loaders and data files were removed once that was established. The one table
below that still cites them is sourced from the retained run summary in
`harness/results/`, not from code you can re-run.

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

## RouterBench results

**Run of 2026-11-18** — [RouterBench](https://huggingface.co/datasets/withmartian/routerbench)
5-shot file (36,483 prompts after dropping 28 ARC-Challenge rows with no grades), replayed
offline by `harness/routerbench.py`. Every number is on a random held-out 30%
(10,956 prompts); the learned checker trains on the other 70%. No model was called:
each tier maps to a RouterBench model and that model's *recorded* score and cost are
read off, so this measures routing decisions, graded and priced with RouterBench's
2023 models rather than this catalog's.

Two ladders:

| ladder | cheap | middle | frontier |
|---|---|---|---|
| A | Mistral-7B | Mixtral-8x7B | GPT-4 |
| B | Mixtral-8x7B | GPT-3.5 | GPT-4 |

The 5-shot file matches the paper's per-model scores (GPT-4 MMLU 82.5% vs 82.8%
reported); the 0-shot file does not (Mistral-7B MMLU 25% — format failures), so earlier
0-shot runs are superseded.

### Without a checker, the router is "always middle"

At the catalog's λ the decision router sends 98% of prompts (10,752 / 10,956) to the
middle tier and never chooses the frontier directly:

| arm | ladder A accuracy | A $/query | ladder B accuracy | B $/query |
|---|---|---|---|---|
| always-cheap | 49.5% | $0.000139 | 64.5% | $0.000415 |
| always-middle | 64.5% | $0.000415 | 68.0% | $0.000710 |
| always-frontier | 79.6% | $0.007970 | 79.6% | $0.007970 |
| **routed, no checker** | **64.1%** | **$0.000412** | **68.0%** | **$0.000707** |
| oracle (ladder) | 79.3% | $0.001081 | 81.9% | $0.001030 |

The oracle — escalate exactly the prompts the middle tier gets wrong — matches
always-frontier at ~1/8 of its cost. That gap is entirely "notice the middle answer is
wrong", which is what the escalation checkers try to do.

### With a checker: middle → frontier escalation

Each checker scores the middle answer's P(correct); below a threshold the frontier
answers instead. Checkers compared:

| checker | what it reads | marginal cost |
|---|---|---|
| learned | MiniLM embeddings of (prompt, answer) → classifier | $0 (local) |
| consistency[cheap] | agreement with the ladder's cheap model | one cheap call |
| consistency[peers] | agreement with 2 similar-priced non-ladder models | ~two middle calls |
| learned + consistency[peers] | mean of the two | ~two middle calls |
| perfect | the recorded grade (ceiling, not achievable) | $0 |

RouterBench records one answer per model, so same-model resampling (`SC_SOURCE=middle`,
the production default) cannot be replayed; cross-model agreement is the proxy.

**How well each checker separates right from wrong middle answers** (test AUC, 0.5 = chance):

| checker | ladder A | ladder B |
|---|---|---|
| learned | 0.709 | 0.709 |
| consistency[cheap] | 0.630 | 0.700 |
| consistency[peers] | 0.770 | 0.750 |
| **learned + consistency[peers]** | **0.813** | **0.807** |

**Compared with no checker** — accuracy and cost at the catalog λ, at the most accurate
threshold costing at most half of always-frontier (chosen on the test split, so these
rows flatter slightly):

| ladder A | accuracy | Δ vs no checker | $/query | cost vs no checker |
|---|---|---|---|---|
| routed, no checker | 64.1% | — | $0.000412 | 1.0× |
| + learned @0.65 | 71.1% | +7.0pp | $0.003262 | 7.9× |
| + consistency[cheap] @0.05 | 73.6% | +9.5pp | $0.004468 | 10.8× |
| + consistency[peers] @0.05 | 72.4% | +8.3pp | $0.003588 | 8.7× |
| + learned + consistency[peers] @0.50 | 73.3% | +9.2pp | $0.003905 | 9.5× |
| + perfect checker | 82.3% | +18.2pp | $0.003169 | 7.7× |

| ladder B | accuracy | Δ vs no checker | $/query | cost vs no checker |
|---|---|---|---|---|
| routed, no checker | 68.0% | — | $0.000707 | 1.0× |
| + learned @0.70 | 74.1% | +6.1pp | $0.003474 | 4.9× |
| + consistency[cheap] @0.05 | 76.2% | +8.2pp | $0.003956 | 5.6× |
| + consistency[peers] @0.05 | 73.7% | +5.7pp | $0.003700 | 5.2× |
| + learned + consistency[peers] @0.45 | 74.1% | +6.1pp | $0.003747 | 5.3× |
| + perfect checker | 82.7% | +14.7pp | $0.002975 | 4.2× |

Every checker buys accuracy, but at a fixed operating point that only says "spending
more on GPT-4 helps". The threshold-free comparison is below.

### AIQ: does a checker beat just picking a model?

AIQ is RouterBench's metric: the mean height of a router's non-decreasing convex hull of
(cost, accuracy) points across the ladder's cost range, swept over λ and checker
threshold. The bar to clear is the paper's **Zero router** — the best mix of fixed
models with no per-prompt routing. "Cost to 95%" is what a router must spend per query
to reach 95% of always-frontier accuracy (75.6%), as a share of always-frontier's cost.

| router | AIQ A | Δ vs no checker | cost to 95%, A | AIQ B | Δ vs no checker | cost to 95%, B |
|---|---|---|---|---|---|---|
| zero (3 ladder models) | 0.7152 | | 75% | 0.7352 | | 69% |
| **zero (all 11 models)** — the bar | **0.7432** | | **59%** | **0.7495** | | **59%** |
| decision, no checker | 0.7152 | — | 75% | 0.7352 | — | 69% |
| + learned | 0.7256 | +0.0104 | 65% | 0.7465 | +0.0113 | 55% |
| + consistency[cheap] | 0.7230 | +0.0078 | 67% | **0.7506** | **+0.0154** | **46%** |
| + consistency[peers] | 0.7291 | +0.0139 | 62% | 0.7425 | +0.0073 | 60% |
| + learned + consistency[peers] | 0.7307 | +0.0154 | 59% | 0.7468 | +0.0116 | 54% |
| + perfect checker | 0.7892 | +0.0740 | 26% | 0.8019 | +0.0667 | 23% |

What this says:

- **Without a checker the router adds nothing.** Its AIQ equals the 3-model Zero router
  exactly on both ladders: sweeping λ moves only a handful of lookups, so it never leaves
  the ladder's own hull. It also sits below the 11-model Zero router — a fixed choice
  from a wider menu beats it.
- **Every checker improves on no checker, by +0.007 to +0.015 AIQ.** Better detection
  does not translate proportionally: the best separator (learned + peers, AUC 0.81)
  pays ~two middle calls on *every* middle-routed prompt, which eats most of its gain.
- **Only one configuration reaches the bar:** ladder B with consistency[cheap] —
  0.7506 vs 0.7495, and 95% of frontier accuracy at 46% of frontier cost vs the Zero
  router's 59%. It works there because ladder B's cheap tier (Mixtral) is strong enough
  that agreeing with it means something, and cheap enough to ask on every prompt. On
  ladder A the same checker is the weakest (cheap tier Mistral-7B, AUC 0.63).
  A +0.001 margin from one split with no confidence interval is a tie, not a win.
- **The ceiling is far away.** A perfect checker clears the bar by +0.05 AIQ and reaches
  95% of frontier at a quarter of its cost. Closing that gap is a checker-quality
  problem, not a routing-logic problem.

This matches the RouterBench paper's own finding: its learned routers (KNN, MLP)
match but do not clearly beat the Zero router, and cascades only pay when the judge's
error rate is around 10% or less.

### Not yet measured

- **Jev** as the checker (`--jev-verify N`) — needs an OpenRouter key. It is the one
  checker that reads the answer against the question rather than inferring difficulty
  or agreement.
- **Same-model self-consistency** (`SC_SOURCE=middle`) — not replayable from RouterBench.
- **This catalog's models.** All of the above uses RouterBench's 2023 ladder; a
  checker trained on Mixtral's answers says nothing about gpt-oss-120b.

### Reproduce

```bash
uv run --extra ml --with pandas harness/routerbench.py --data routerbench_5shot.pkl \
  --out harness/results/rb5sc_mistral_ladder.json
uv run --extra ml --with pandas harness/routerbench.py --data routerbench_5shot.pkl \
  --cheap mistralai/mixtral-8x7b-chat --middle gpt-3.5-turbo-1106 --frontier gpt-4-1106-preview \
  --out harness/results/rb5sc_gpt35_ladder.json
```

Seed 42, embeddings of each prompt's last 1,000 characters (5-shot prompts put the
question last; MiniLM reads 256 tokens from the start).

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

**Check the dataset reproduces its paper before using it.** RouterBench's 0-shot file
scores Mistral-7B at 25% on MMLU (random for four options) where the paper reports 56%;
the 5-shot file matches. Two full runs on the 0-shot file had to be discarded.

**Unwrap recorded answers before parsing them.** RouterBench stores answers as
`str(list)` (`"['A\\n']"`), so letter extraction silently found nothing until the
wrapper was stripped. Answers starting "A man walks…" were also read as choice A, and
code was compared by its last number.

**A checker's curve must include "checker off".** Without the un-escalated points, a
checker that costs money on every prompt can score below the router it is attached to,
which is not a real outcome — the operator can always switch it off.

**Cost accounting must include failures.** An arm that errors more will look cheaper if
errored prompts are dropped from the totals.

---

*Last reviewed: 2026-11-18. Raw results in `harness/results/` (gitignored); every
figure above is reproducible from the commands shown.*
