# SmartRoute

### Cut LLM costs by 40%+ without sacrificing quality. Coming soon.

SmartRoute is an intelligent inference router. It classifies every prompt as EASY, MEDIUM, or HARD, sends easy prompts to cheap models, uses a cross-provider verifier to catch mistakes before you see them, and only reaches for frontier models when they're actually needed.

**One endpoint. Every provider. Always the cheapest model that can do the job.**

- **Smart routing** — difficulty classifier assigns every prompt to the right tier
- **Quality guardrail** — cross-provider verifier rejects weak cheap-model answers before returning
- **Frontier on demand** — complex problems still get top-tier reasoning
- **Full transparency** — every request returns a `RoutingDecision` with cost, latency, and cascade path

---

Public release, benchmark numbers, and the hosted API are all on the way.  
Watch this repo or reach out at kartikay3@outlook.com if you want early access.
