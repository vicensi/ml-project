"""Deterministic synthetic hourly demand generator."""

from tsplatform.data_generation.generator import (
    SeriesProfile,
    build_dataset_info,
    generate_dataset,
    make_profile,
)
from tsplatform.data_generation.segments import Segment, compute_segments

__all__ = [
    "Segment",
    "SeriesProfile",
    "build_dataset_info",
    "compute_segments",
    "generate_dataset",
    "make_profile",
]
