.PHONY: install lint test cov notebooks docker doctor

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

notebooks:
	@for nb in notebooks/tutorial_*.ipynb; do \
		echo "Running $$nb..."; \
		poetry run jupyter nbconvert --to notebook --execute "$$nb" \
			--ExecutePreprocessor.timeout=120 \
			--output /tmp/executed_$$(basename "$$nb") || exit 1; \
	done

docker:
	docker build -f deployment/Dockerfile -t deepiri-fuselk:local .

doctor:
	poetry run fuselk doctor
