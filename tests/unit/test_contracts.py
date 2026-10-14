from datetime import datetime, timezone

import pandas as pd
import pytest
from pydantic import ValidationError

from tsplatform.contracts.raw import RawObservation, validate_raw_frame


def _frame(**over):
    base = pd.DataFrame({
        "series_id": ["S001", "S001"],
        "event_timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
        "target_value": [1.0, 2.0],
        "is_anomaly": [False, False],
    })
    for k, v in over.items():
        base[k] = v
    return base


def test_valid_frame_passes():
    validate_raw_frame(_frame())


@pytest.mark.parametrize("bad", [
    {"target_value": [1.0, -1.0]},
    {"target_value": [1.0, None]},
    {"event_timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30"])},
    {"event_timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:00"])},
    {"event_timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]).tz_localize("UTC")},
])
def test_invalid_frames_rejected(bad):
    with pytest.raises(ValueError):
        validate_raw_frame(_frame(**bad))


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        validate_raw_frame(_frame().drop(columns=["is_anomaly"]))


def test_raw_observation_model():
    ok = RawObservation(series_id="S001", event_timestamp=datetime(2025, 1, 1, 3), target_value=3.2)
    assert ok.is_anomaly is False
    with pytest.raises(ValidationError):
        RawObservation(series_id="S001", event_timestamp=datetime(2025, 1, 1, 3, tzinfo=timezone.utc), target_value=1)
    with pytest.raises(ValidationError):
        RawObservation(series_id="", event_timestamp=datetime(2025, 1, 1), target_value=1)
    with pytest.raises(ValidationError):
        RawObservation(series_id="S", event_timestamp=datetime(2025, 1, 1), target_value=float("nan"))
