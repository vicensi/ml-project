"""Chronological segments of the synthetic timeline.

Boundaries are snapped to midnight so a day never straddles two segments. Intervals are
half-open: [start, end). The *production* segment is held back: Phase 4/5 replays it hour by
hour to simulate live inferences and the late arrival of actual values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from tsplatform.config.settings import GeneratorConfig

SEGMENT_NAMES = ("train", "validation", "test", "production")


@dataclass(frozen=True)
class Segment:
    name: str
    start: datetime  # inclusive
    end: datetime  # exclusive


def compute_segments(cfg: GeneratorConfig) -> dict[str, Segment]:
    start = pd.Timestamp(cfg.start)
    end = start + pd.Timedelta(hours=cfg.n_hours)  # exclusive end of the whole timeline
    total_days = (end - start) / pd.Timedelta(days=1)

    fractions = [getattr(cfg.segments, name) for name in SEGMENT_NAMES]
    boundaries = [start]
    acc = 0.0
    for frac in fractions[:-1]:
        acc += frac
        cut = (start + pd.Timedelta(days=total_days * acc)).floor("D")
        cut = max(cut, boundaries[-1] + pd.Timedelta(days=1))
        boundaries.append(cut)
    boundaries.append(end)

    segs = {
        name: Segment(name, boundaries[i].to_pydatetime(), boundaries[i + 1].to_pydatetime())
        for i, name in enumerate(SEGMENT_NAMES)
    }
    for s in segs.values():
        if s.end <= s.start:
            raise ValueError(f"segment '{s.name}' is empty; increase n_hours")
    return segs


def assign_segment(df: pd.DataFrame, segs: dict[str, Segment]) -> pd.Series:
    """Label each row with its segment name (useful for tests and the Streamlit explorer)."""
    out = pd.Series(index=df.index, dtype="object")
    for s in segs.values():
        out[(df["event_timestamp"] >= s.start) & (df["event_timestamp"] < s.end)] = s.name
    return out
