"""Repositories: the only place that issues SQL for raw data and the dataset registry.

Idempotency: writes are UPSERTs keyed by the natural primary key, so re-running an ingestion
(Airflow retry, manual re-run) never duplicates rows.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from tsplatform.contracts.features import FEATURE_TABLE_COLUMNS, validate_feature_frame
from tsplatform.contracts.raw import RAW_COLUMNS, DatasetInfo, validate_raw_frame
from tsplatform.data_access.tables import dataset_registry, raw_observations, series_features


def _insert_for(engine: Engine):
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    elif engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    else:  # pragma: no cover
        raise NotImplementedError(f"upsert not implemented for {engine.dialect.name}")
    return insert


def _chunks(records: list[dict], size: int) -> Iterator[list[dict]]:
    for i in range(0, len(records), size):
        yield records[i : i + size]


_MAX_PARAMS = 30000  # below SQLite's 32766 and Postgres' 65535 bind-parameter limits


def _upsert_rows(engine: Engine, table, rows: list[dict], keys: list[str], updates: list[str]) -> None:
    """One transaction, chunked multi-row INSERT ... ON CONFLICT DO UPDATE."""
    if not rows:
        return
    insert = _insert_for(engine)
    chunk_size = max(1, _MAX_PARAMS // len(rows[0]))
    with engine.begin() as conn:
        for chunk in _chunks(rows, chunk_size):
            stmt = insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                index_elements=keys, set_={c: stmt.excluded[c] for c in updates}
            )
            conn.execute(stmt)


class RawObservationRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def upsert(self, df: pd.DataFrame, dataset_id: str) -> int:
        """Insert or update rows; returns the number of input rows processed."""
        validate_raw_frame(df)
        # to_dict boxes values into native Python types (pd.Timestamp is a datetime subclass)
        rows = [{**r, "dataset_id": dataset_id} for r in df[RAW_COLUMNS].to_dict("records")]
        _upsert_rows(
            self.engine, raw_observations, rows,
            keys=["series_id", "event_timestamp"],
            updates=["target_value", "is_anomaly", "dataset_id"],
        )
        return len(rows)

    def read(
        self,
        series_ids: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Half-open interval [start, end)."""
        t = raw_observations
        stmt = select(t.c.series_id, t.c.event_timestamp, t.c.target_value, t.c.is_anomaly)
        if series_ids:
            stmt = stmt.where(t.c.series_id.in_(series_ids))
        if start is not None:
            stmt = stmt.where(t.c.event_timestamp >= start)
        if end is not None:
            stmt = stmt.where(t.c.event_timestamp < end)
        stmt = stmt.order_by(t.c.series_id, t.c.event_timestamp)
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        return df

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(raw_observations)).scalar_one())

    def series_ids(self) -> list[str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(raw_observations.c.series_id).distinct().order_by(raw_observations.c.series_id)
            )
            return [r[0] for r in rows]


class DatasetRegistry:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def register(self, info: DatasetInfo) -> None:
        insert = _insert_for(self.engine)
        values = info.model_dump()
        stmt = insert(dataset_registry).values(**values)
        stmt = stmt.on_conflict_do_nothing(index_elements=["dataset_id"])  # same id => same data
        with self.engine.begin() as conn:
            conn.execute(stmt)

    def get(self, dataset_id: str) -> DatasetInfo | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(dataset_registry).where(dataset_registry.c.dataset_id == dataset_id)
            ).mappings().first()
        if row is None:
            return None
        return DatasetInfo(**{k: row[k] for k in DatasetInfo.model_fields})


class FeatureRepository:
    """Offline feature table (source of the Feast FeatureView)."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def upsert(self, df: pd.DataFrame) -> int:
        validate_feature_frame(df)
        rows = df[FEATURE_TABLE_COLUMNS].to_dict("records")
        _upsert_rows(
            self.engine, series_features, rows,
            keys=["series_id", "event_timestamp"],
            updates=[c for c in FEATURE_TABLE_COLUMNS if c not in ("series_id", "event_timestamp")],
        )
        return len(rows)

    def read(self, series_ids: list[str] | None = None) -> pd.DataFrame:
        t = series_features
        stmt = select(*[t.c[c] for c in FEATURE_TABLE_COLUMNS])
        if series_ids:
            stmt = stmt.where(t.c.series_id.in_(series_ids))
        stmt = stmt.order_by(t.c.series_id, t.c.event_timestamp)
        with self.engine.connect() as conn:
            df = pd.read_sql(stmt, conn)
        df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
        return df

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(series_features)).scalar_one())

    def max_timestamp(self) -> datetime | None:
        with self.engine.connect() as conn:
            ts = conn.execute(select(func.max(series_features.c.event_timestamp))).scalar_one()
        return pd.Timestamp(ts).to_pydatetime() if ts is not None else None
