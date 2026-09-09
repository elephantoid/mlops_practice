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

COPY pyproject.toml /tmp/churnwatch/pyproject.toml

USER airflow

# The base image runs Airflow out of a virtualenv at /home/airflow/.local, created with
# include-system-site-packages = false. /usr/local/lib/python3.12/site-packages is
# therefore NOT on sys.path, and `uv pip install --system` -- as root or otherwise --
# installs into a directory nothing ever reads: the build succeeds and the first task
# fails with ModuleNotFoundError. Name that venv's interpreter explicitly instead.
ENV AIRFLOW_VENV_PYTHON=/home/airflow/.local/bin/python

# Dependencies come from pyproject.toml rather than a hand-copied list: a second list
# would drift from the real one silently, and the DAG would fail at import time in a
# container instead of at review time.
#
# apache-airflow is pinned into the SAME resolution on purpose. Installing the project
# deps alone lets the resolver satisfy mlflow or pandas by moving Airflow's own pins,
# producing an image where the scheduler no longer starts. Naming Airflow here makes any
# such conflict a loud build failure instead.
RUN uv pip install --python "${AIRFLOW_VENV_PYTHON}" --no-cache \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/churnwatch/pyproject.toml

# Fail the build rather than the first DAG run if the resolution above quietly dropped
# something. Run through the venv interpreter, as the scheduler does.
RUN "${AIRFLOW_VENV_PYTHON}" -c \
    "import airflow, mlflow, lightgbm, evidently; print('airflow', airflow.__version__)"
