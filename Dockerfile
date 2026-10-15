# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src
WORKDIR /app
COPY requirements/base.txt requirements/base.txt
RUN pip install -r requirements/base.txt
COPY pyproject.toml ./
COPY src ./src
RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app
CMD ["python", "-m", "tsplatform.cli", "--help"]

# Phase 2: + Feast. Used by the feature jobs; connection settings arrive via environment variables.
FROM base AS features
USER root
COPY requirements/features.txt requirements/features.txt
RUN pip install -r requirements/features.txt
COPY feast_repo ./feast_repo
RUN chown -R app /app
ENV TSP_FEAST_REPO_PATH=/app/feast_repo
USER app

FROM features AS dev
USER root
COPY requirements/dev.txt requirements/dev.txt
RUN pip install -r requirements/dev.txt
COPY tests ./tests
RUN chown -R app /app
USER app
CMD ["pytest", "-q"]
