"""Engine factory and schema bootstrap.

SQLite (unit tests only) has no schemas, so ``raw.`` is translated away. Postgres is the
only supported runtime database.
"""

from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from tsplatform.config.settings import DatabaseSettings
from tsplatform.data_access.tables import FEATURES_SCHEMA, RAW_SCHEMA, metadata


def make_engine(db: DatabaseSettings, **kwargs) -> Engine:
    url = db.sqlalchemy_url()
    engine = create_engine(url, pool_pre_ping=True, future=True, **kwargs)
    if engine.dialect.name == "sqlite":
        engine = engine.execution_options(schema_translate_map={RAW_SCHEMA: None, FEATURES_SCHEMA: None})
    return engine


def create_app_schema(engine: Engine) -> None:
    """Idempotent: CREATE SCHEMA IF NOT EXISTS + CREATE TABLE IF NOT EXISTS."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            for schema in (RAW_SCHEMA, FEATURES_SCHEMA):
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    metadata.create_all(engine, checkfirst=True)
