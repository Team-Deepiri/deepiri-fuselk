FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --only main

ENV PYTHONPATH=/app/src

CMD ["fuselk", "doctor"]
