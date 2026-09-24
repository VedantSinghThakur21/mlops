# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Dockerfile for the Housing Price Prediction FastAPI inference service.
#
# Multi-stage build:
#   1. "builder" installs Python dependencies into a virtualenv.
#   2. Final stage is a minimal runtime image containing only the venv,
#      the app code, and the trained model artifacts (model.pkl + scaler.pkl)
#      baked in — no external model download/network call is needed at
#      container start, so it deploys cleanly on an EC2 instance (plain
#      Docker, ECS-on-EC2, or behind an ALB) even with restricted egress.
# ---------------------------------------------------------------------------

# ---------- Stage 1: build dependencies -------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Build tools for any package that needs to compile from source (e.g. scipy
# wheels not published for the target platform).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime image ------------------------------------------
FROM python:3.11-slim AS runtime

# Minimal runtime packages: curl only for the container HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (good practice for any host, including EC2).
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

# Bring in the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Application code.
COPY app/ ./app/

# Trained model artifacts, baked into the image so the container is
# self-contained at runtime.
COPY models/ ./models/
COPY data/features/scaler.pkl ./data/features/scaler.pkl

# Writable log directory for the app's file logging.
RUN mkdir -p /app/logs && chown -R app:app /app

USER app

EXPOSE 8000

# Used by Docker/ECS to determine container health; also works behind an
# EC2 Application Load Balancer target-group health check on the same path.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker by default; scale horizontally (multiple EC2 tasks/instances
# behind a load balancer) rather than forking many workers per container.
# Override WEB_CONCURRENCY at `docker run` time to change this.
ENV WEB_CONCURRENCY=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
