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

from tsplatform.contracts.raw import RAW_COLUMNS, DatasetInfo, validate_raw_frame
from tsplatform.data_access.tables import dataset_registry, raw_observations


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


class RawObservationRepository:
    def __init__(self, engine: Engine, chunk_size: int = 5000) -> None:
        self.engine = engine
        self.chunk_size = chunk_size

    def upsert(self, df: pd.DataFrame, dataset_id: str) -> int:
        """Insert or update rows; returns the number of input rows processed."""
        validate_raw_frame(df)
        insert = _insert_for(self.engine)
        # to_dict boxes values into native Python types (pd.Timestamp is a datetime subclass)
        rows = [{**r, "dataset_id": dataset_id} for r in df[RAW_COLUMNS].to_dict("records")]
        with self.engine.begin() as conn:  # single transaction: all-or-nothing
            for chunk in _chunks(rows, self.chunk_size):
                stmt = insert(raw_observations).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["series_id", "event_timestamp"],
                    set_={
                        "target_value": stmt.excluded.target_value,
                        "is_anomaly": stmt.excluded.is_anomaly,
                        "dataset_id": stmt.excluded.dataset_id,
                    },
                )
                conn.execute(stmt)
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
