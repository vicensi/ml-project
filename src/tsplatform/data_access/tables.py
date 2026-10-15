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
FEATURES_SCHEMA = "features"

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

# Offline source of the Feast FeatureView. Written by the feature build step (idempotent upsert);
# Feast reads it with point-in-time joins. Only complete rows exist (see contracts/features.py).
series_features = Table(
    "series_features",
    metadata,
    Column("series_id", String(64), primary_key=True),
    Column("event_timestamp", DateTime, primary_key=True),
    Column("value_lag_1", Float, nullable=False),
    Column("value_lag_2", Float, nullable=False),
    Column("value_lag_24", Float, nullable=False),
    Column("value_lag_168", Float, nullable=False),
    Column("rolling_mean_24", Float, nullable=False),
    Column("rolling_std_24", Float, nullable=False),
    Column("rolling_mean_168", Float, nullable=False),
    Column("hour_of_day", Integer, nullable=False),
    Column("day_of_week", Integer, nullable=False),
    schema=FEATURES_SCHEMA,
)
