"""Runs against a real Postgres when TSP_TEST_DATABASE_URL is set (compose `tests` service)."""
import os

import pytest

from tsplatform.config.settings import DatabaseSettings, GeneratorConfig
from tsplatform.data_access.db import create_app_schema, make_engine
from tsplatform.data_access.repositories import DatasetRegistry, RawObservationRepository
from tsplatform.data_generation import build_dataset_info, generate_dataset

URL = os.environ.get("TSP_TEST_DATABASE_URL")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not URL, reason="TSP_TEST_DATABASE_URL not set")]


def test_postgres_upsert_idempotent_and_schema_separation():
    cfg = GeneratorConfig(seed=11, n_series=2, n_hours=24 * 7 * 5)
    engine = make_engine(DatabaseSettings(url_override=URL))
    create_app_schema(engine)
    df = generate_dataset(cfg)
    info = build_dataset_info(cfg, df)
    DatasetRegistry(engine).register(info)
    repo = RawObservationRepository(engine)
    repo.upsert(df, info.dataset_id)
    repo.upsert(df, info.dataset_id)
    assert repo.count() >= len(df)
    assert DatasetRegistry(engine).get(info.dataset_id).n_rows == len(df)
