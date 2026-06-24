.PHONY: install lint test cov benchmark docker doctor notebooks

install:
	poetry install --no-interaction

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests
	poetry run mypy src/deepiri_fuselk

test:
	poetry run pytest tests/ -m "not slow and not gpu" -q

cov:
	poetry run pytest tests/ -m "not slow and not gpu" --cov=deepiri_fuselk --cov-report=term-missing

benchmark:
	poetry run python scripts/benchmark.py --all

notebooks:
	poetry run python scripts/generate_notebooks.py

docker:
	docker build -f deployment/Dockerfile -t deepiri-fuselk:local .

doctor:
	poetry run fuselk doctor

gui:
	poetry run fuselk gui
