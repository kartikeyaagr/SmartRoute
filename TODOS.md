# SmartRoute — Roadmap

Items intentionally excluded from v0 to keep scope small and prove the core thesis first.
Graduate these once benchmark numbers validate the approach.

---

## v0.1 — Production API Layer

> **Why this milestone bundle:** An API server is the prerequisite for everything that follows (cache, observability, multi-tenant). Auth and Postgres logging are bundled here because they sit on the same request-path middleware chain — splitting them into a separate milestone doubles integration work and leaves the API in an unusable (unauthenticated, unlogged) state in between.

- [x] FastAPI server wrapping `Router.route_async` — `src/smartroute/server.py`
- [x] `POST /v1/chat/completions` — OpenAI-compatible endpoint
- [x] `X-SmartRoute-Meta` response header with routing metadata (JSON-encoded)
- [x] Health endpoint (`GET /health`)
- [x] SSE fake-streaming support (`stream: true`)
- [x] API key auth middleware (`SMARTROUTE_SERVER_API_KEY`, Bearer token)
- [x] Docker + docker-compose
- [ ] Request/response logging to Postgres

---

## v0.2 — Semantic Prompt Cache

> **Why this milestone bundle:** Cache benefits multiply with traffic. Getting the cache architecture right (pluggable interface, per-tenant scoping, TTL policy) before traffic exists means no painful retrofit later. Ships in two stages: (1) pluggable interface + in-memory exact-match cache to validate the seam, (2) pgvector semantic cache once the seam is proven.

### Stage 1 — Cache seam (COMPLETE ✅)

- [x] `src/smartroute/cache.py` — `PromptCache` protocol + `NoOpCache` + `InMemoryLRUCache` (exact-match)
- [x] `RoutingDecision` adds `cache_hit: bool` + `cache_similarity: float | None`
- [x] `Router.route_async` checks cache before classifier; writes on successful LLM call
- [x] `CACHE_BACKEND` setting (`none` | `memory` | `pgvector`, default `none`)
- [x] `CACHE_SIMILARITY_THRESHOLD` + `CACHE_TTL_SECONDS` settings
- [x] Tests: cache miss → unchanged behavior; cache hit → skips providers; put/get roundtrip

### Stage 2 — pgvector semantic cache

- [ ] `PgvectorCache` implementation — embed prompt → cosine similarity lookup
- [ ] Cache TTL sweep + invalidation policy
- [ ] Cache metrics: hit rate, cost saved, latency improvement
- [ ] Cache must be per-tenant-scoped (critical — cross-tenant hits = privacy breach)

---

## v0.3 — Observability

> **Why this milestone bundle:** Observability is a prerequisite for enterprise sales (customers want dashboards before they write checks) and for catching production regressions. All four items (Prometheus, Grafana, structured logs, OTel traces) use the same instrumentation layer — building them separately adds more overhead than shipping together.

- [ ] Prometheus metrics: latency histograms, cost counters, escalation rate, cache hit rate
- [ ] Grafana dashboard
- [ ] Structured logging (JSON) with trace IDs
- [ ] OpenTelemetry traces across provider calls

---

## v0.4 — Classifier Improvements

> **Why this milestone:** The keyword classifier is a cold-start heuristic. After v0.1–v0.3 collect production routing decisions (RoutingDecision JSONL), those decisions become supervision data for a fine-tuned classifier. This is the data flywheel that turns accumulated traffic into a moat.

- [ ] Fine-tune classifier on MMLU routing outcomes (use RoutingDecision JSONL as supervision)
- [ ] Per-domain difficulty priors (law prompts → bias toward HARD)
- [ ] User feedback loop: thumbs up/down → update routing weights
- [ ] A/B test Keyword vs DeBERTa classifier on same prompt set

---

## v0.5 — Production Hardening

> **Why this milestone:** Single-instance reliability is table stakes for paying customers. Redis and circuit breakers become necessary the moment a second API server instance is running (shared cache locks, coordinated failover). Bundle with rate limiting and graceful degradation because they're all part of the same "what happens when a provider goes down" answer.

- [ ] Redis distributed lock for cache writes (multi-instance safe)
- [ ] Circuit breaker per provider (stop routing to degraded models)
- [ ] Rate limit budgeting across providers
- [ ] Graceful degradation: if all cheap models down, route everything to frontier
- [ ] Async model health checks with exponential backoff

---

## v0.6 — Multi-Tenant Foundation

> **Why this milestone:** Multi-tenancy unlocks the private beta with ICP #1. Without it, every customer shares one routing config and one cost bucket — unusable for production. Per-tenant isolation (separate DB schema + separate cache namespace) is a prerequisite for everything that follows.

- [ ] Per-tenant API key management
- [ ] Per-tenant routing config (model preferences, tier overrides)
- [ ] Per-tenant cost allocation and usage metering
- [ ] Per-tenant cache namespace (prevents cross-tenant cache hits)
- [ ] Rate limiter per tenant (RPS + daily $ cap)

---

## v0.7 — Billing Integration

> **Why this milestone:** Revenue. Free tier is fine for community, but the metered billing layer must be live before the public launch. Stripe metered billing is the standard for infra-SaaS.

- [ ] Stripe metered billing integration
- [ ] Free tier limits enforcement (10k req/mo cap)
- [ ] Pro / Scale tier gating
- [ ] Usage dashboard in customer portal (Stripe billing portal + custom overlay)

---

## v0.8 — Customer Dashboard

> **Why this milestone:** The "aha" metric ("you saved $X this month vs always-frontier") must be visible in the first session after signup. Without a dashboard, customers can't see the value they're getting — which means they churn even when the product is working. Dashboard gates paid conversion.

- [ ] Next.js frontend: API key management, usage graphs, cost-per-tier breakdown
- [ ] Hero metric: "$ saved vs always-frontier this month"
- [ ] Cascade-path histogram, verifier stats, per-model savings breakdown
- [ ] Policy controls UI (provider pinning, tier overrides)
- [ ] Downloadable audit JSONL (for compliance review)

---

## v0.9 — Compliance & Policy

> **Why this milestone:** Required before any enterprise POC. SOC 2 Type I process, DPA templates, and audit logs are non-negotiable for companies that have gone through a security review.

- [ ] SOC 2 Type I audit initiated
- [ ] DPA / SCC templates for EU customers
- [ ] Regional data residency options (US / EU)
- [ ] Audit log API (read-only, scoped per tenant)
- [ ] Prompt retention controls (opt-in, configurable window)

---

## Longer Term (v1.0+)

- [ ] Streaming responses (`/v1/chat/completions` with `stream=true`)
- [ ] Custom cheap/frontier model lists via config (not hardcoded)
- [ ] Verifier fine-tuning: train on (prompt, response, correct_answer) tuples from benchmark
- [ ] Per-request and per-tenant daily spend limits
- [ ] Self-hosted frontier option: vLLM on A100 as frontier fallback
- [ ] Enterprise self-hosted image (for HIPAA/FedRAMP workloads)
