"""Feature construction.

Everything is derived from ``y.shift(k)`` with k >= 1 on a *calendar-complete* hourly index, so
"row stamped t" can only see observations stamped < t. Rolling statistics are computed on the
already-shifted series ``prev = y.shift(1)``; the window ending at row t therefore covers
y[t-w .. t-1] and never y[t].

Gaps: the index is reindexed to every hour between the first and last observation. A missing
hour becomes NaN, any feature whose window touches it becomes NaN, and that row is dropped.
Nothing is interpolated or forward-filled (that would invent or leak information).

``include_next_hour=True`` also emits the row stamped ``last_observed + 1h``: this is the
serving row ("what we know right now, to predict the next hour"); it has no target yet.
"""

from __future__ import annotations

import pandas as pd

from tsplatform.contracts.features import (
    ENTITY_KEY,
    FEATURE_NAMES,
    FEATURE_TABLE_COLUMNS,
    LAGS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
)
from tsplatform.contracts.raw import validate_raw_frame

_HOUR = pd.Timedelta(hours=1)


def _series_features(g: pd.DataFrame, include_next_hour: bool) -> pd.DataFrame:
    y = g.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN].sort_index()
    end = y.index.max() + (_HOUR if include_next_hour else pd.Timedelta(0))
    y = y.reindex(pd.date_range(y.index.min(), end, freq="h"))  # gaps -> NaN, never filled

    out = pd.DataFrame(index=y.index)
    for k in LAGS:
        out[f"value_lag_{k}"] = y.shift(k)

    prev = y.shift(1)  # window for row t ends at t-1
    out["rolling_mean_24"] = prev.rolling(ROLLING_WINDOWS["rolling_mean_24"], min_periods=24).mean()
    out["rolling_std_24"] = prev.rolling(ROLLING_WINDOWS["rolling_std_24"], min_periods=24).std()
    out["rolling_mean_168"] = prev.rolling(ROLLING_WINDOWS["rolling_mean_168"], min_periods=168).mean()

    out["hour_of_day"] = out.index.hour.astype("int32")  # known in advance: no leakage
    out["day_of_week"] = out.index.dayofweek.astype("int32")

    out = out.dropna(subset=[n for n in FEATURE_NAMES if n not in ("hour_of_day", "day_of_week")])
    out.index.name = TIMESTAMP_COLUMN
    out = out.reset_index()
    out.insert(0, ENTITY_KEY, g[ENTITY_KEY].iloc[0])
    return out


def build_features(raw: pd.DataFrame, include_next_hour: bool = True) -> pd.DataFrame:
    """Raw hourly history -> complete feature rows, sorted by (series_id, event_timestamp)."""
    validate_raw_frame(raw)
    parts = [_series_features(g, include_next_hour) for _, g in raw.groupby(ENTITY_KEY, sort=True)]
    df = pd.concat(parts, ignore_index=True)[FEATURE_TABLE_COLUMNS]
    for c in FEATURE_NAMES:
        df[c] = df[c].astype("int32" if c in ("hour_of_day", "day_of_week") else "float64")
    return df


def build_targets(raw: pd.DataFrame) -> pd.DataFrame:
    """Entity frame for training: (series_id, t, y_t). y_t is the value observed *at* t, i.e. the
    value that the feature row stamped t (built from data < t) must predict."""
    validate_raw_frame(raw)
    return raw[[ENTITY_KEY, TIMESTAMP_COLUMN, TARGET_COLUMN]].sort_values(
        [ENTITY_KEY, TIMESTAMP_COLUMN]
    ).reset_index(drop=True)
