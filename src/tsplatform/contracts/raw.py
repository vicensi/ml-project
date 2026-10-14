"""Contract: generator -> persistence (raw hourly history).

Columns
  series_id        str       entity key (Feast join key later)
  event_timestamp  datetime  naive UTC, aligned to the hour
  target_value     float     observed demand, >= 0
  is_anomaly       bool      ground-truth label of injected events. NEVER a model feature.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, field_validator

RAW_COLUMNS = ["series_id", "event_timestamp", "target_value", "is_anomaly"]


class RawObservation(BaseModel):
    series_id: str = Field(min_length=1, max_length=64)
    event_timestamp: datetime
    target_value: float = Field(ge=0.0, allow_inf_nan=False)
    is_anomaly: bool = False

    @field_validator("event_timestamp")
    @classmethod
    def _hour_aligned_naive(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("event_timestamp must be naive UTC")
        if v.minute or v.second or v.microsecond:
            raise ValueError("event_timestamp must be aligned to the hour")
        return v


class DatasetInfo(BaseModel):
    """Identity of a generated dataset; stored in the registry and later logged to MLflow."""

    dataset_id: str
    config_hash: str
    data_hash: str
    config_json: str
    n_rows: int
    n_series: int
    ts_min: datetime
    ts_max: datetime


def validate_raw_frame(df: pd.DataFrame) -> None:
    """Structural validation (cheap, vectorised). Raises ValueError with a precise reason."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df.empty:
        raise ValueError("raw frame is empty")
    if not pd.api.types.is_datetime64_any_dtype(df["event_timestamp"]):
        raise ValueError("event_timestamp must be datetime64")
    if getattr(df["event_timestamp"].dt, "tz", None) is not None:
        raise ValueError("event_timestamp must be naive UTC")
    if (df["event_timestamp"].dt.floor("h") != df["event_timestamp"]).any():
        raise ValueError("event_timestamp must be aligned to the hour")
    if df[["series_id", "event_timestamp", "target_value"]].isna().any().any():
        raise ValueError("nulls in series_id/event_timestamp/target_value")
    if (df["target_value"] < 0).any():
        raise ValueError("target_value must be >= 0")
    if df.duplicated(["series_id", "event_timestamp"]).any():
        raise ValueError("duplicate (series_id, event_timestamp) rows")
