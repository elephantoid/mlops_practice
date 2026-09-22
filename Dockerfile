# syntax=docker/dockerfile:1
#
# Build from the repo root, after exporting the model:
#
#   uv run python -m src.models.export
#   docker build -t riskwatch:latest \
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


# Development stage. Deliberately placed BEFORE runtime: a multi-stage build defaults to
# its LAST stage, so runtime must stay last or the documented
# `docker build -t riskwatch:latest .` would start producing a dev image.
#
# No source is COPYed. docker-compose.yml bind-mounts the working tree at the same absolute
# path the host uses, so edits are live and PROJECT_ROOT -- which every module derives from
# its own file location, not the CWD -- resolves identically on both sides. That also keeps
# the absolute artifact paths MLflow writes into mlflow.db valid from either side.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS dev

# The project venv must not live in the mounted tree. `.venv` sits inside the repo and holds
# macOS arm64 wheels: unusable here, and uv would find it first. UV_PROJECT_ENVIRONMENT moves
# this container's venv out of the bind mount's reach.
# UV_COMPILE_BYTECODE is deliberately NOT here. As a persistent ENV it makes uv recompile
# every .pyc on each `uv run` -- measured at 400-800ms of pure overhead per command, on top
# of an environment that is already complete. It is set on the sync below instead, so the
# bytecode is built once into the image and simply reused.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# libgomp1 for LightGBM, exactly as in the runtime stage -- this is the Linux OpenMP runtime,
# not the macOS `brew install libomp` fix. git and make because pre-commit and the Makefile
# are part of the development loop; curl to poke the API from inside the container.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 git make curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --dev, unlike the builder stage: pytest, ruff and pre-commit are the entire point here.
RUN --mount=type=cache,target=/opt/uv-cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    UV_COMPILE_BYTECODE=1 uv sync --locked --dev --no-install-project

# A login shell sources /etc/profile, which overwrites PATH outright and would drop
# /opt/venv/bin -- leaving `python` as the interpreter with none of the dependencies, silently.
# Interactive non-login shells (what `make shell` gives) keep the ENV above, but VS Code
# terminals and `su -` do not, so put it back where a login shell will find it.
RUN printf 'PATH="/opt/venv/bin:$PATH"\n' > /etc/profile.d/10-venv.sh

# Runs as root. On a macOS bind mount ownership is remapped to the invoking user, so matching
# uids buys nothing, and re-syncing /opt/venv needs write access.
CMD ["bash"]


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

RUN useradd --create-home --uid 10001 riskwatch

WORKDIR /app

COPY --from=builder --chown=riskwatch:riskwatch /app/.venv /app/.venv
COPY --chown=riskwatch:riskwatch src/ /app/src/
COPY --chown=riskwatch:riskwatch build/model/ /app/model/

# Must exist and be owned before dropping privileges: the process runs as uid 10001 and
# cannot create or write into a root-owned directory. Without this the prediction log
# silently fails on every request -- it is written defensively and never raises.
RUN mkdir -p /app/logs && chown riskwatch:riskwatch /app/logs

USER riskwatch

EXPOSE 8000

# urllib rather than curl, so the image needs no extra package for the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
