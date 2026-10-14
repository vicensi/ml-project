import pandas as pd
import pytest

from tsplatform.config.settings import GeneratorConfig, SegmentConfig
from tsplatform.data_generation import compute_segments, generate_dataset
from tsplatform.data_generation.segments import assign_segment


def test_segments_are_contiguous_ordered_and_cover_timeline(small_cfg):
    segs = compute_segments(small_cfg)
    names = list(segs)
    assert names == ["train", "validation", "test", "production"]
    for a, b in zip(names, names[1:]):
        assert segs[a].end == segs[b].start
        assert segs[a].start < segs[a].end
    assert segs["train"].start == small_cfg.start
    assert segs["production"].end == pd.Timestamp(small_cfg.start) + pd.Timedelta(hours=small_cfg.n_hours)


def test_every_row_belongs_to_exactly_one_segment(small_cfg):
    df = generate_dataset(small_cfg)
    labels = assign_segment(df, compute_segments(small_cfg))
    assert labels.notna().all()
    assert set(labels.unique()) == {"train", "validation", "test", "production"}


def test_segment_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SegmentConfig(train=0.5, validation=0.2, test=0.1, production=0.1)
