"""Table definitions. Postgres schemas separate concerns inside the application database:

  raw         immutable-ish history written by the generator/ingestion
  serving     predictions + actuals (Phase 4)
  monitoring  Evidently runs and alerts (Phase 5)

Airflow and MLflow use their *own databases* (see docker/postgres/init), not these schemas.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    func,
)

metadata = MetaData()

RAW_SCHEMA = "raw"

raw_observations = Table(
    "observations",
    metadata,
    Column("series_id", String(64), primary_key=True),
    Column("event_timestamp", DateTime, primary_key=True),
    Column("target_value", Float, nullable=False),
    Column("is_anomaly", Boolean, nullable=False, default=False),
    Column("dataset_id", String(64), nullable=False),
    Column("ingested_at", DateTime, nullable=False, server_default=func.now()),
    schema=RAW_SCHEMA,
)

dataset_registry = Table(
    "datasets",
    metadata,
    Column("dataset_id", String(64), primary_key=True),
    Column("config_hash", String(64), nullable=False),
    Column("data_hash", String(64), nullable=False),
    Column("config_json", Text, nullable=False),
    Column("n_rows", Integer, nullable=False),
    Column("n_series", Integer, nullable=False),
    Column("ts_min", DateTime, nullable=False),
    Column("ts_max", DateTime, nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    schema=RAW_SCHEMA,
)
