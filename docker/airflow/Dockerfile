# syntax=docker/dockerfile:1
#
# Airflow with the ChurnWatch dependencies, for the Milestone 4 retraining DAG.
#
#   docker compose -f docker-compose.yml -f docker-compose.airflow.yml build
#
# The DAG calls src/ functions in-process rather than shelling out, so this image needs
# everything the training and monitoring code imports -- MLflow, LightGBM, Evidently.
# The repo itself is NOT copied in: it is bind-mounted at /opt/airflow/project so DAG and
# src edits take effect without a rebuild.

ARG AIRFLOW_VERSION=2.10.5
FROM apache/airflow:${AIRFLOW_VERSION}-python3.12

ARG AIRFLOW_VERSION

USER root

# LightGBM's Linux wheel links against libgomp. Same reasoning as the serving Dockerfile:
# this is the Linux OpenMP runtime, NOT the macOS `brew install libomp` fix, which must
# never appear in an image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv rather than pip -- project-wide constraint, and it resolves this dependency set in
# seconds rather than minutes.
COPY --from=ghcr.io/astral-sh/uv:0.9.7 /uv /usr/local/bin/uv

# uv.lock, not just pyproject.toml. The serving image builds with `uv sync --locked`, and
# this one resolving open-ended ranges on every build broke that guarantee: a rebuild could
# silently pick up a different MLflow, Evidently or LightGBM, invalidating the image that
# was actually verified. uv.lock is committed deliberately for exactly this reason.
# --chown because the install below runs as `airflow`: a root-owned directory here
# fails with EACCES the moment uv tries to write its exported requirements into it.
COPY --chown=airflow:root pyproject.toml uv.lock /tmp/churnwatch/

USER airflow

# The base image runs Airflow out of a virtualenv at /home/airflow/.local, created with
# include-system-site-packages = false. /usr/local/lib/python3.12/site-packages is
# therefore NOT on sys.path, and `uv pip install --system` -- as root or otherwise --
# installs into a directory nothing ever reads: the build succeeds and the first task
# fails with ModuleNotFoundError. Name that venv's interpreter explicitly instead.
ENV AIRFLOW_VENV_PYTHON=/home/airflow/.local/bin/python

# Two steps, because the two halves need different guarantees.
#
# First the project's dependencies, exported from uv.lock as fully pinned requirements --
# every version identical to the one `make check` ran against, and to the serving image.
# `uv export` reads the lockfile; it does not re-resolve.
#
# Then Airflow itself, pinned, installed into the same environment afterwards. Airflow ships
# its own constraints and would otherwise be free to move a shared dependency to satisfy
# them; installing it second means any genuine incompatibility surfaces here as a build
# failure rather than as a scheduler that will not start.
RUN uv export --project /tmp/churnwatch --no-dev --no-emit-project \
        --format requirements-txt --frozen --output-file /tmp/churnwatch/requirements.txt \
    && uv pip install --python "${AIRFLOW_VENV_PYTHON}" --no-cache \
        -r /tmp/churnwatch/requirements.txt \
    && uv pip install --python "${AIRFLOW_VENV_PYTHON}" --no-cache \
        "apache-airflow==${AIRFLOW_VERSION}"

# Fail the build rather than the first DAG run if the resolution above quietly dropped
# something. Run through the venv interpreter, as the scheduler does.
RUN "${AIRFLOW_VENV_PYTHON}" -c \
    "import airflow, mlflow, lightgbm, evidently; print('airflow', airflow.__version__)"
