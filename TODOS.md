# SmartRoute — Roadmap

> Single source of truth for project state and roadmap.
> Update after every feature add or completed task.

**Current status:** v0.1 complete — 75 tests passing. Starting v0.2 pgvector cache.

---

## v0 — Library (COMPLETE ✅)

- [x] Foundation: `config.py`, `providers.py`, `__init__.py`, `.env.example`
- [x] Classifier: `KeywordBaselineClassifier` + `DeBERTaNLIClassifier`
- [x] MMLU data loader: `bench/data/mmlu_500.jsonl` (500 prompts, EASY 36.6% / MEDIUM 43.0% / HARD 20.4%)
- [x] Verifier + Router: full cascade logic, RoutingDecision dataclass, JSONL logging
- [x] Benchmark runner: all 4 ablation modes, resume, dry-run, cost ceiling
- [x] Report: confusion matrix, markdown table, JSON output
- [x] README, public API exports
- [x] Cache seam: `NoOpCache` + `InMemoryLRUCache` (exact-match LRU, pluggable `PromptCache` protocol)

---

## Pending: run the benchmark

Needs real API keys in `.env`.

```bash
cp .env.example .env   # fill in keys
uv run bench/run_benchmark.py --mode full --prompts 50   # smoke run (~$0.03)
uv run bench/run_benchmark.py --all-modes                # full 500-prompt run
uv run bench/report.py
```

Success criteria (if <25% cost reduction, reassess routing thresholds before v0.1):
- Full SmartRoute ≥90% MMLU accuracy vs always-frontier
- ≥40% cost reduction vs always-frontier
- p95 latency ≤1.5× always-frontier
- Extraction failure rate <5%

---

## v0.1 — Production API Layer (COMPLETE ✅)

> **Why this milestone bundle:** API server is the prerequisite for everything that follows. Auth and Postgres logging sit on the same request-path middleware chain — splitting them doubles integration work.

- [x] FastAPI server wrapping `Router.route_async` (`src/smartroute/server.py`)
- [x] `POST /v1/chat/completions` — OpenAI-compatible endpoint
- [x] `X-SmartRoute-Meta` response header with routing metadata (JSON-encoded)
- [x] Health endpoint (`GET /health`)
- [x] SSE fake-streaming support (`stream: true`)
- [x] API key auth middleware (`SMARTROUTE_SERVER_API_KEY`, Bearer token, constant-time compare)
- [x] Server tests: 7 tests (health, success, ProviderError, auth permutations)
- [x] Docker: `Dockerfile` + `docker-compose.yml`
- [x] Request/response logging to Postgres (`src/smartroute/db.py`, `routing_decisions` table)
- [x] Postgres service in docker-compose.yml (pgvector/pgvector:pg17, healthcheck)

---

## v0.2 — Semantic Prompt Cache

> **Why this milestone bundle:** Cache architecture (pluggable interface, per-tenant scoping, TTL) must be right before traffic exists — painful to retrofit. Stage 1 already done (in-memory exact-match seam). Stage 2 adds pgvector semantic similarity.

### Stage 1 — Cache seam (COMPLETE ✅)

- [x] `PromptCache` protocol + `NoOpCache` + `InMemoryLRUCache`
- [x] `RoutingDecision` adds `cache_hit` + `cache_similarity` fields
- [x] `Router.route_async` checks cache before classifier; writes on success
- [x] `CACHE_BACKEND` / `CACHE_TTL_SECONDS` / `CACHE_MAX_SIZE` settings

### Stage 2 — pgvector semantic cache

- [ ] `PgvectorCache` — embed prompt → cosine similarity lookup
- [ ] Per-tenant cache namespace (cross-tenant hits = privacy breach — non-negotiable)
- [ ] Cache TTL sweep + invalidation
- [ ] Cache metrics: hit rate, cost saved, latency delta

---

## v0.3 — Observability

> **Why this milestone bundle:** Observability is a prerequisite for enterprise sales and catching production regressions. All four items share the same instrumentation layer.

- [ ] Prometheus metrics: latency histograms, cost counters, escalation rate, cache hit rate
- [ ] Grafana dashboard
- [ ] Structured JSON logging with trace IDs
- [ ] OpenTelemetry traces across provider calls

---

## v0.4 — Classifier Improvements

> **Why this milestone:** Keyword classifier is a cold-start heuristic. RoutingDecision JSONL from v0.1–v0.3 becomes supervision data for a fine-tuned classifier — the data flywheel that builds the moat.

- [ ] Fine-tune classifier on MMLU routing outcomes (RoutingDecision JSONL as supervision)
- [ ] Per-domain difficulty priors (law prompts → bias toward HARD)
- [ ] User feedback loop: thumbs up/down → update routing weights
- [ ] A/B test Keyword vs DeBERTa on same prompt set

---

## v0.5 — Production Hardening

> **Why this milestone:** Redis and circuit breakers become necessary the moment a second API server instance is running.

- [ ] Redis distributed lock for cache writes (multi-instance safe)
- [ ] Circuit breaker per provider (stop routing to degraded models)
- [ ] Rate limit budgeting across providers
- [ ] Graceful degradation: all cheap models down → route everything to frontier
- [ ] Async model health checks with exponential backoff

---

## v0.6 — Multi-Tenant Foundation

> **Why this milestone:** Unlocks private beta with ICP #1. Without it, every customer shares one routing config and one cost bucket.

- [ ] Per-tenant API key management
- [ ] Per-tenant routing config (model preferences, tier overrides)
- [ ] Per-tenant cost allocation and usage metering
- [ ] Per-tenant cache namespace (prevents cross-tenant cache hits)
- [ ] Rate limiter per tenant (RPS + daily $ cap)

---

## v0.7 — Billing Integration

- [ ] Stripe metered billing integration
- [ ] Free tier enforcement (10k req/mo cap)
- [ ] Pro / Scale tier gating
- [ ] Usage dashboard in customer portal

---

## v0.8 — Customer Dashboard

> **Why this milestone:** "$ saved vs always-frontier this month" must be visible in first session — it gates paid conversion.

- [ ] Next.js frontend: API key management, usage graphs, cost-per-tier breakdown
- [ ] Hero metric: "$ saved vs always-frontier this month"
- [ ] Cascade-path histogram, verifier stats, per-model savings breakdown
- [ ] Policy controls UI (provider pinning, tier overrides)
- [ ] Downloadable audit JSONL (for compliance review)

---

## v0.9 — Compliance & Policy

- [ ] SOC 2 Type I audit initiated
- [ ] DPA / SCC templates for EU customers
- [ ] Regional data residency options (US / EU)
- [ ] Audit log API (read-only, scoped per tenant)
- [ ] Prompt retention controls (opt-in, configurable window)

---

## v1.0+ — Long Term

- [ ] True streaming responses (not fake-SSE)
- [ ] Custom cheap/frontier model lists via config (not hardcoded)
- [ ] Verifier fine-tuning on benchmark outcomes
- [ ] Per-request and per-tenant daily spend limits
- [ ] Self-hosted frontier option: vLLM on A100 as frontier fallback
- [ ] Enterprise self-hosted image (for HIPAA/FedRAMP workloads)
