.PHONY: clean check lint fmt shell

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
