# Contributing to deepiri-fuselk

## Workflow

1. Branch from `dev`: `git checkout -b feat/your-feature dev`
2. Make changes with tests
3. Run `make lint && make test`
4. Open PR to `dev` (not `main`)
5. Tag `@Team-Deepiri/support-team`
6. Update Plaky to "Needs QA"

## Code style

- Python 3.11+
- Ruff for lint/format (line length 100)
- Mypy on `src/deepiri_fuselk`
- pytest for all new functionality

## Commit conventions

- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring
- `docs:` documentation
- `test:` tests only
- `ci:` CI/CD changes
