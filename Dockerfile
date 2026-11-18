FROM python:3.13-slim

WORKDIR /app

# Pinned, not :latest — an unpinned build tool makes the same commit build differently.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

# Dependencies first, in their own layer: code edits don't reinstall torch.
# The ml extra is required in production — without it the decision layer silently
# falls back to the keyword heuristic (see harness/eval_routing.py: 32% of hard prompts
# routed to the cheap tier vs 1.7% with the trained triage). torch is CPU-only on Linux
# (pyproject [tool.uv.sources]).
COPY pyproject.toml uv.lock LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project --extra server --extra ml

# Bake the MiniLM encoder into the image so pods never download it at startup.
ENV HF_HOME=/app/.cache/huggingface
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY src/ ./src/
# Default catalog: catalog.py resolves models.yaml relative to src/ (i.e. /app/models.yaml).
# Kubernetes mounts its own over CATALOG_PATH; this is the fallback.
COPY models.yaml ./
RUN uv sync --frozen --no-dev --extra server --extra ml

RUN useradd --uid 10001 --no-create-home app && chown -R app /app/.cache
USER 10001

ENV PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=1
EXPOSE 8000

CMD ["smartroute-server"]
