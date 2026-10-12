.PHONY: clean check lint fmt shell image

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache  -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

check:
	uv run pytest tests/ -v; ec=$$?; [ $$ec -eq 0 ] || [ $$ec -eq 5 ]

lint:
	uv run ruff check src/ tests/ dags/
	uv run ruff format --check src/ tests/ dags/

fmt:
	uv run ruff check --fix src/ tests/ dags/
	uv run ruff format src/ tests/ dags/

# Drop into the Linux development container. Everything above runs unchanged in there --
# each target goes through `uv run`, which works the same on either side -- so this only
# changes where you are standing, not what you type. Deliberately not wired into check/lint:
# those would then recurse once you are already inside.
shell:
	docker compose run --rm dev bash

# Export both tracks' production models, then build the serving image around them. One target so
# the export cannot be forgotten: a skipped export bakes whatever build/model/ last held. Each
# export replaces its own build/model/<track>/ and the Dockerfile copies only those two.
# For Cloud Run: make image PLATFORM=linux/amd64
IMAGE ?= riskwatch:latest
PLATFORM ?=
image:
	uv run python -m src.models.export --track credit
	uv run python -m src.models.export --track fraud
	docker build $(if $(PLATFORM),--platform $(PLATFORM)) -t $(IMAGE) .
