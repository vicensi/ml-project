"""Training frame via Feast point-in-time join.

For every observation (series, t, y_t) of a chronological segment we ask Feast: "what did the
feature view say about this series as of t?" and attach y_t as the label. Splitting is done by
*segment boundaries on t* (train < validation < test), never at random.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from tsplatform.contracts.features import (
    ENTITY_KEY,
    FEATURE_NAMES,
    FEATURE_SERVICE_NAME,
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    TrainingFrameInfo,
    feature_set_id,
)
from tsplatform.data_access.repositories import RawObservationRepository
from tsplatform.data_generation.segments import Segment
from tsplatform.feature_store.client import historical_features
from tsplatform.features import build_targets


def build_training_frame(
    store,
    raw_repo: RawObservationRepository,
    segment: Segment,
    series_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, TrainingFrameInfo]:
    raw = raw_repo.read(series_ids=series_ids, start=segment.start, end=segment.end)
    entity_df = build_targets(raw)
    joined = historical_features(store, entity_df)

    complete = joined[FEATURE_NAMES].notna().all(axis=1)
    frame = (
        joined[complete]
        .sort_values([ENTITY_KEY, TIMESTAMP_COLUMN])
        .reset_index(drop=True)[[ENTITY_KEY, TIMESTAMP_COLUMN, *FEATURE_NAMES, TARGET_COLUMN]]
    )
    if frame.empty:
        raise ValueError(f"no complete training rows in segment '{segment.name}'")
    if not (frame[TIMESTAMP_COLUMN] < segment.end).all() or not (frame[TIMESTAMP_COLUMN] >= segment.start).all():
        raise AssertionError("training frame escaped its segment boundaries")  # defensive

    frame_hash = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
    info = TrainingFrameInfo(
        feature_set_id=feature_set_id(),
        feature_service=FEATURE_SERVICE_NAME,
        segment=segment.name,
        segment_start=segment.start,
        segment_end=segment.end,
        n_entity_rows=len(entity_df),
        n_rows=len(frame),
        n_dropped_incomplete=int((~complete).sum()),
        ts_min=frame[TIMESTAMP_COLUMN].min().to_pydatetime(),
        ts_max=frame[TIMESTAMP_COLUMN].max().to_pydatetime(),
        frame_hash=frame_hash,
    )
    return frame, info
