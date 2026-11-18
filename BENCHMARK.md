# The SmartRoute benchmark

How to measure whether routing saves you money, what the harness measures, and what it
found.

- [Why this exists](#why-this-exists) — what a routing benchmark has to prove
- [Running it](#running-it) — the three commands
- [The corpus](#the-corpus) — 200 questions, and why not MMLU
- [The arms](#the-arms) — what each one isolates
- [Metrics](#metrics) — what every column means
- [Results](#results) — what was measured, 2026-09-21
- [RouterBench results](#routerbench-results) — 36k prompts: where routing *does* pay, 2026-11-18
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

> **TL;DR** — On 36,483 RouterBench prompts, the two-layer router alone is no better
> than always using the middle model. Adding a **Jev answer check** (escalate to the
> frontier model when Jev doubts the middle model's answer) reaches **95% of GPT-4's
> accuracy at 23–28% of GPT-4's cost**, and beats RouterBench's Zero-router baseline on
> both model ladders tested, with 95% confidence intervals that exclude zero.

**Run of 2026-11-18.** [RouterBench](https://huggingface.co/datasets/withmartian/routerbench)
5-shot file: 36,483 prompts across MMLU, HellaSwag, GSM8K, ARC, Winogrande, MBPP,
MT-Bench and smaller sets (28 rows without grades dropped). Replayed offline by
`harness/routerbench.py`: each routing tier maps to a RouterBench model, and that
model's *recorded* grade and cost are read off. Only the Jev checks make live calls.

**Protocol.** 5 random 70/30 splits. The learned checker trains on each 70%; every
number is measured on the held-out 30% (~10,956 prompts) and averaged over the 5 splits.
Brackets are 95% confidence intervals, corrected for overlapping test sets
(Nadeau-Bengio). "Δ" columns are paired differences within each split.

| ladder | cheap | middle | frontier |
|---|---|---|---|
| **A** | Mistral-7B | Mixtral-8x7B | GPT-4 |
| **B** | Mixtral-8x7B | GPT-3.5 | GPT-4 |

### 1. The router alone is "always middle"

At the catalog's cost-of-a-wrong-answer setting, the router sends 98% of prompts to the
middle tier and never chooses the frontier directly (split 1 shown):

| | ladder A accuracy | $/query | ladder B accuracy | $/query |
|---|---|---|---|---|
| always cheap | 49.5% | $0.00014 | 64.5% | $0.00042 |
| always middle | 64.5% | $0.00042 | 68.0% | $0.00071 |
| always frontier | 79.6% | $0.00797 | 79.6% | $0.00797 |
| **router, no checker** | **64.1%** | **$0.00041** | **68.0%** | **$0.00071** |
| oracle: escalate exactly the wrong answers | 79.3% | $0.00108 | 81.9% | $0.00103 |

The oracle matches GPT-4 at about 1/8 of its cost. All of that gap is one problem:
**noticing when the middle model's answer is wrong.** So the experiment is about the
checker that decides when to escalate.

### 2. Checkers: how well each one spots a wrong answer

AUC of each checker's P(correct) against the recorded grade (0.5 = coin flip, 1.0 = perfect):

| checker | how it works | extra cost / prompt | AUC, ladder A | AUC, ladder B |
|---|---|---|---|---|
| learned | classifier on MiniLM embeddings of prompt + answer | $0 | 0.700 [0.689, 0.712] | 0.707 [0.695, 0.720] |
| agreement, cheap | does the cheap model give the same answer? | 1 cheap call | 0.633 [0.621, 0.645] | 0.694 [0.683, 0.705] |
| agreement, peers | do 2 similar-priced models agree? | ~2 middle calls | 0.773 [0.766, 0.781] | 0.748 [0.740, 0.757] |
| learned + peers | average of the two | ~2 middle calls | 0.814 [0.809, 0.819] | 0.804 [0.794, 0.815] |
| **Jev** | decision model asked "is this answer correct?" | **~$0.000045** | **0.933 [0.927, 0.938]** | **0.938 [0.933, 0.944]** |

Combining Jev with any other checker lowered its AUC (0.83–0.93), so Jev is used alone.

### 3. Does it beat just picking a model?

**AIQ** is RouterBench's metric: the average accuracy a router achieves across the whole
cost range from always-cheap to always-frontier (higher is better). The bar is
RouterBench's **Zero router** — the best fixed model, or mix of fixed models, at each
price, with no per-prompt routing at all, chosen from all 11 models.

| router | AIQ, A | Δ vs Zero router | AIQ, B | Δ vs Zero router |
|---|---|---|---|---|
| Zero router (bar to beat) | 0.750 | — | 0.757 | — |
| router, no checker | 0.720 | −0.030 [−0.034, −0.026] | 0.741 | −0.015 [−0.020, −0.011] |
| + learned + peers | 0.736 | −0.014 [−0.017, −0.011] | 0.752 | −0.004 [−0.008, −0.001] |
| + agreement, cheap | 0.728 | −0.022 [−0.027, −0.016] | 0.756 | −0.000 [−0.006, +0.005] |
| **+ Jev** | **0.772** | **+0.022 [+0.019, +0.026]** | **0.788** | **+0.032 [+0.028, +0.035]** |
| + perfect checker (ceiling) | 0.794 | +0.044 | 0.808 | +0.051 |

**Cost to reach 95% of GPT-4's accuracy**, as a share of always using GPT-4:

| router | ladder A | ladder B |
|---|---|---|
| router, no checker | 75% | 69% |
| Zero router | 58% | 58% |
| + learned + peers | 59% | 55% |
| **+ Jev** | **27.5%** | **22.7%** |
| + perfect checker | 26% | 23% |

One operating point from each ladder (Jev threshold 0.65 and 0.60; split 1):

| | accuracy | $/query | vs always-GPT-4 |
|---|---|---|---|
| always GPT-4 | 79.6% | $0.00797 | — |
| ladder A + Jev | 79.2% | $0.00363 | −0.4pp at 46% of the cost |
| ladder B + Jev | 80.0% | $0.00322 | +0.4pp at 40% of the cost |

### What this shows

- **Routing logic was never the bottleneck; error detection was.** Without a checker
  the router sits below the Zero router on both ladders. With a perfect checker it
  clears it easily. Every improvement measured here came from the checker.
- **Jev nearly closes the gap to a perfect checker** — 95% of GPT-4 accuracy at 27.5% vs
  26% (A) and 22.7% vs 23% (B) of GPT-4's cost — because it is both accurate (AUC 0.93)
  and cheap (~6–11% of a middle-model call).
- **Cheaper signals do not get there.** Agreement between models is a real signal
  (AUC up to 0.81 combined), but asking two more models on every prompt costs more than
  the escalations it saves. The one previously reported "win" (ladder B, cheap-model
  agreement, +0.001 on a single split) is a tie once intervals are computed.
- **Consistent with the RouterBench paper**, whose learned routers match but do not
  beat the Zero router and whose cascades only pay with a judge error rate near 10%.

### Caveats

- **Benchmark contamination.** MMLU, HellaSwag and GSM8K are public and very likely in
  Jev's training data, so part of its 0.93 AUC may be memorised answers rather than
  judgement. Expect it to be weaker on novel traffic; measure it on your own graded
  answers before relying on these numbers.
- **2023 models.** RouterBench's ladder is Mistral / Mixtral / GPT-3.5 / GPT-4, not this
  repo's catalog. The results test the routing and escalation decisions, not the
  shipped models.
- **Same-model self-consistency** (`SC_SOURCE=middle`) cannot be replayed — RouterBench
  records one answer per model — so only cross-model agreement was measured.
- **Jev pricing.** Calls went to TypeSafe's API, which reports tokens but not cost; they
  are priced at Jev 1.14's OpenRouter list price ($0.04198 per million input tokens).
  The full run was 72,966 checks, about $3.25, with zero failures.
- Example operating points use thresholds picked on the test split; AIQ and
  cost-to-95% do not depend on a threshold.

### Reproduce

```bash
export TYPESAFE_API_KEY=...        # or OPENROUTER_API_KEY; Jev scores are cached
uv run --extra ml --with pandas harness/routerbench.py --data routerbench_5shot.pkl \
  --splits 5 --jev --out harness/results/rbci_mistral_ladder.json
uv run --extra ml --with pandas harness/routerbench.py --data routerbench_5shot.pkl \
  --splits 5 --jev --cheap mistralai/mixtral-8x7b-chat --middle gpt-3.5-turbo-1106 \
  --frontier gpt-4-1106-preview --out harness/results/rbci_gpt35_ladder.json
```

Seed 42. Embeddings use each prompt's last 1,000 characters (5-shot prompts put the
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
