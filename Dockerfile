FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install with server extras only (no ML/bench)
RUN uv sync --frozen --extra server --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["smartroute-server"]
