"""Contract: features <-> Feast <-> training <-> serving.

TIME SEMANTICS (the single most important rule of this project)
  A feature row stamped ``t`` is the *state of knowledge at the start of hour t*: it is computed
  only from observations with ``event_timestamp < t``. The target associated with that row is
  the observation stamped ``t`` itself (the "next value" after the last one known).

  - training:  entity row (series, t, y_t)  --PIT join-->  feature row with greatest stamp <= t
  - serving:   latest materialised row for the series (stamp <= "now")
  Both therefore read the *same* definition, and neither can see y_t or anything later.

Rows are emitted only when every feature is computable (>= 168 hours of contiguous history);
a missing row therefore means "insufficient history" — values are never imputed or back-filled.
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta

import pandas as pd
from datetime import datetime
from pydantic import BaseModel

FEATURE_SPEC_VERSION = "v1"
FEATURE_VIEW_NAME = "series_features"
FEATURE_SERVICE_NAME = f"forecast_features_{FEATURE_SPEC_VERSION}"
ENTITY_NAME = "series"
ENTITY_KEY = "series_id"
TIMESTAMP_COLUMN = "event_timestamp"
TARGET_COLUMN = "target_value"

LAGS = (1, 2, 24, 168)
ROLLING_WINDOWS = {"rolling_mean_24": 24, "rolling_std_24": 24, "rolling_mean_168": 168}
MIN_HISTORY_HOURS = 168

FLOAT_FEATURES = [
    "value_lag_1",
    "value_lag_2",
    "value_lag_24",
    "value_lag_168",
    "rolling_mean_24",
    "rolling_std_24",
    "rolling_mean_168",
]
INT_FEATURES = ["hour_of_day", "day_of_week"]
FEATURE_NAMES = FLOAT_FEATURES + INT_FEATURES
FEATURE_TABLE_COLUMNS = [ENTITY_KEY, TIMESTAMP_COLUMN, *FEATURE_NAMES]

# A row older than this is considered stale: for 1-step-ahead hourly forecasts, features that
# are 6+ hours old mean the feature pipeline is down, so the model must not silently use them.
FEATURE_TTL = timedelta(hours=6)


def feature_set_id() -> str:
    """Identifier of the feature definition (names, lags, windows, ttl). Logged to MLflow."""
    spec = {
        "version": FEATURE_SPEC_VERSION,
        "features": FEATURE_NAMES,
        "lags": LAGS,
        "rolling": ROLLING_WINDOWS,
        "ttl_hours": FEATURE_TTL.total_seconds() / 3600,
    }
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return f"fs-{FEATURE_SPEC_VERSION}-{digest}"


def feature_refs() -> list[str]:
    return [f"{FEATURE_VIEW_NAME}:{name}" for name in FEATURE_NAMES]


def validate_feature_frame(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURE_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    if df.empty:
        raise ValueError("feature frame is empty")
    if df[FEATURE_TABLE_COLUMNS].isna().any().any():
        raise ValueError("feature rows must be complete (no nulls); incomplete rows are not emitted")
    if df.duplicated([ENTITY_KEY, TIMESTAMP_COLUMN]).any():
        raise ValueError("duplicate (series_id, event_timestamp) feature rows")
    if not df["hour_of_day"].between(0, 23).all() or not df["day_of_week"].between(0, 6).all():
        raise ValueError("calendar features out of range")


class TrainingFrameInfo(BaseModel):
    """Lineage of a point-in-time training frame; logged to MLflow in Phase 3."""

    feature_set_id: str
    feature_service: str
    segment: str
    segment_start: datetime
    segment_end: datetime
    n_entity_rows: int
    n_rows: int
    n_dropped_incomplete: int  # warm-up rows / gaps: no complete feature row existed at t
    ts_min: datetime
    ts_max: datetime
    frame_hash: str
