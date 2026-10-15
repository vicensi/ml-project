"""Thin, testable layer over feast.FeatureStore.

Error mapping (consumed by the API in Phase 4)
  FeatureStoreConfigError      -> infrastructure/config problem (maps to 503)
  FeatureStoreUnavailable      -> Feast/Redis/Postgres call failed (maps to 503)
Business conditions (unknown series, insufficient history) are NOT exceptions here: they are
returned as rows with ``status != "ok"`` so the caller decides (404 / 422).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from tsplatform.contracts.features import (
    ENTITY_KEY,
    FEATURE_NAMES,
    FEATURE_SERVICE_NAME,
    FEATURE_TTL,
    TIMESTAMP_COLUMN,
    feature_refs,
    feature_set_id,
)
from tsplatform.observability.logging import log_operation

log = logging.getLogger(__name__)


class FeatureStoreConfigError(RuntimeError):
    error_category = "infrastructure"


class FeatureStoreUnavailable(RuntimeError):
    error_category = "infrastructure"


def get_store(repo_path: str | os.PathLike):
    """Open the Feast store; fail early if the yaml still contains unresolved ${VARS}."""
    from feast import FeatureStore

    cfg = Path(repo_path) / "feature_store.yaml"
    if not cfg.exists():
        raise FeatureStoreConfigError(f"feature_store.yaml not found in {repo_path}")
    expanded = os.path.expandvars(cfg.read_text())
    if "${" in expanded:
        unresolved = sorted({tok.split("}")[0] for tok in expanded.split("${")[1:]})
        raise FeatureStoreConfigError(f"unresolved environment variables: {unresolved}")
    return FeatureStore(repo_path=str(repo_path))


def apply_definitions(store) -> None:
    """`feast apply`: register entity, source, feature view and feature service."""
    from tsplatform.feature_store.definitions import ALL_OBJECTS

    with log_operation(log, "feast_apply", feature_set_id=feature_set_id()):
        store.apply(ALL_OBJECTS)


def materialize(store, end: datetime, start: datetime | None = None) -> None:
    """Copy the latest offline row per series with stamp in [start, end] to the online store.

    ``end`` is the simulated/real "now": no row stamped after it can reach the online store, which
    keeps serving free of future data.

    ``start=None`` (the normal, idempotent mode):
      * first run   -> start = end - FEATURE_TTL (older rows would be stale anyway). Feast's own
                       default is "now - ttl" in wall-clock time, which is wrong for replayed or
                       historical data — this was caught by the end-to-end CLI run.
      * later runs  -> incremental, continuing from the last interval recorded in the registry.
      * ``end`` not after the last materialised instant -> no-op (the online store never moves back).
    """
    from tsplatform.contracts.features import FEATURE_VIEW_NAME

    end_ts = pd.Timestamp(end)
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    with log_operation(log, "feast_materialize", end=str(end), explicit_start=start is not None) as r:
        if start is not None:
            store.materialize(start_date=start, end_date=end_ts.to_pydatetime())
            return
        intervals = store.get_feature_view(FEATURE_VIEW_NAME).materialization_intervals
        if not intervals:
            store.materialize(start_date=(end_ts - FEATURE_TTL).to_pydatetime(), end_date=end_ts.to_pydatetime())
            return
        last_end = max(pd.Timestamp(e) for _, e in intervals)
        last_end = last_end.tz_localize("UTC") if last_end.tzinfo is None else last_end.tz_convert("UTC")
        if end_ts <= last_end:
            r["skipped"] = f"already materialised up to {last_end}"
            return
        store.materialize_incremental(end_date=end_ts.to_pydatetime())


def historical_features(store, entity_df: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time join. ``entity_df``: series_id, event_timestamp (prediction instant t) and any
    extra pass-through columns (e.g. the target). Returns one row per input row."""
    for col in (ENTITY_KEY, TIMESTAMP_COLUMN):
        if col not in entity_df.columns:
            raise ValueError(f"entity_df is missing '{col}'")
    with log_operation(log, "feast_historical_features", rows=len(entity_df)) as r:
        try:
            df = store.get_historical_features(
                entity_df=entity_df, features=store.get_feature_service(FEATURE_SERVICE_NAME)
            ).to_df()
        except Exception as exc:  # feast raises many types; callers only need the category
            raise FeatureStoreUnavailable(type(exc).__name__) from exc
        r["returned"] = len(df)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], utc=True).dt.tz_localize(None)
    return df


def online_features(store, series_ids: list[str]) -> pd.DataFrame:
    """Latest materialised features per series. Columns: series_id, <features>, status.

    status: "ok" (all features present) | "not_found" (no row online: unknown series, history too
    short, or not materialised yet). Missing features are never filled.
    """
    with log_operation(log, "feast_online_features", series=len(series_ids)):
        try:
            resp = store.get_online_features(
                features=feature_refs(), entity_rows=[{ENTITY_KEY: s} for s in series_ids]
            ).to_dict()
        except Exception as exc:
            raise FeatureStoreUnavailable(type(exc).__name__) from exc
    df = pd.DataFrame({ENTITY_KEY: resp[ENTITY_KEY], **{n: resp[n] for n in FEATURE_NAMES}})
    complete = df[FEATURE_NAMES].notna().all(axis=1)
    df["status"] = complete.map({True: "ok", False: "not_found"})
    return df


def verify_online_availability(store, series_ids: list[str]) -> dict:
    """Task `verify_feature_availability`: every expected series must have complete online features."""
    df = online_features(store, series_ids)
    missing = df.loc[df["status"] != "ok", ENTITY_KEY].tolist()
    return {"checked": len(series_ids), "ok": len(series_ids) - len(missing), "missing": missing}
