# syntax=docker/dockerfile:1
#
# Build from the repo root, after exporting the model:
#
#   uv run python -m src.models.export
#   docker build -t churnwatch:latest \
#       --build-arg MODEL_VERSION="$(cat build/model/MODEL_VERSION)" .
#
# The image is self-contained: the model is baked in, so `docker run` needs no MLflow
# server reachable. docker-compose.yml overrides MODEL_URI to load from the registry
# instead.

# The uv image is built on python:3.12-slim-bookworm, so the virtualenv it produces is
# ABI- and path-compatible with the runtime stage below.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# --no-dev keeps pytest, ruff and pre-commit out of a serving image.
# --no-install-project because pyproject.toml declares no build backend: this is a
# "virtual" project, so src/ is copied directly rather than installed as a package.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project


FROM python:3.12-slim-bookworm AS runtime

ARG MODEL_VERSION=unknown

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_URI=/app/model \
    MODEL_VERSION=${MODEL_VERSION}

# LightGBM's Linux wheel links against libgomp, which slim images do not ship. This is
# the Linux OpenMP runtime -- not the macOS `brew install libomp` fix, which must never
# appear here.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 churnwatch

WORKDIR /app

COPY --from=builder --chown=churnwatch:churnwatch /app/.venv /app/.venv
COPY --chown=churnwatch:churnwatch src/ /app/src/
COPY --chown=churnwatch:churnwatch build/model/ /app/model/

# Must exist and be owned before dropping privileges: the process runs as uid 10001 and
# cannot create or write into a root-owned directory. Without this the prediction log
# silently fails on every request -- it is written defensively and never raises.
RUN mkdir -p /app/logs && chown churnwatch:churnwatch /app/logs

USER churnwatch

EXPOSE 8000

# urllib rather than curl, so the image needs no extra package for the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
