"""Data contracts shared between components (Pydantic models + frame-level validators)."""

from tsplatform.contracts.features import (
    FEATURE_NAMES,
    FEATURE_SERVICE_NAME,
    TrainingFrameInfo,
    feature_set_id,
    validate_feature_frame,
)
from tsplatform.contracts.raw import (
    RAW_COLUMNS,
    DatasetInfo,
    RawObservation,
    validate_raw_frame,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SERVICE_NAME",
    "RAW_COLUMNS",
    "DatasetInfo",
    "RawObservation",
    "TrainingFrameInfo",
    "feature_set_id",
    "validate_feature_frame",
    "validate_raw_frame",
]
