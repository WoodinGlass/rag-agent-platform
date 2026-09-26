# syntax=docker/dockerfile:1.7
#
# Multi-stage build:
#   - builder: install deps + package into a venv
#   - runtime: slim image, copies venv + app, non-root user
#
# Build:  docker build -f docker/Dockerfile -t rag-agent-platform:dev .
# Run:    docker run --rm -p 8000:8000 --env-file .env rag-agent-platform:dev

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy everything the install needs — no partial-file tricks
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY evals/ ./evals/

ARG EXTRAS="http,vector"
RUN pip install --upgrade pip && \
    pip install ".[${EXTRAS}]"

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    LOG_JSON=true \
    LOG_LEVEL=INFO \
    VECTOR_BACKEND=memory \
    EMBEDDER_PROVIDER=fake \
    LLM_PROVIDER=fake

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/app /app/app
COPY --from=builder /build/scripts /app/scripts
COPY --from=builder /build/evals /app/evals
COPY --from=builder /build/pyproject.toml /app/pyproject.toml
COPY --from=builder /build/README.md /app/README.md

WORKDIR /app
RUN chown -R app:app /app

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/healthz" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
