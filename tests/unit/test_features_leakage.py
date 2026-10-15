"""Temporal-correctness tests. These are the proof that features never see the future."""
import numpy as np
import pandas as pd
import pytest

from tsplatform.config.settings import GeneratorConfig
from tsplatform.contracts.features import (
    FEATURE_NAMES,
    FEATURE_TABLE_COLUMNS,
    feature_set_id,
    validate_feature_frame,
)
from tsplatform.data_generation import generate_dataset
from tsplatform.features import build_features, build_targets

H = pd.Timedelta(hours=1)


@pytest.fixture(scope="module")
def raw():
    return generate_dataset(GeneratorConfig(seed=5, n_series=3, n_hours=24 * 7 * 5, anomaly_rate=0.01))


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def _y(raw, sid):
    g = raw[raw.series_id == sid].set_index("event_timestamp")["target_value"]
    return g


def test_columns_and_contract(feats):
    assert list(feats.columns) == FEATURE_TABLE_COLUMNS
    validate_feature_frame(feats)


def test_lag_semantics_row_t_uses_only_values_before_t(raw, feats):
    y = _y(raw, "S001")
    f = feats[feats.series_id == "S001"].set_index("event_timestamp")
    for t in f.index[::37]:
        assert f.loc[t, "value_lag_1"] == y[t - 1 * H]
        assert f.loc[t, "value_lag_2"] == y[t - 2 * H]
        assert f.loc[t, "value_lag_24"] == y[t - 24 * H]
        assert f.loc[t, "value_lag_168"] == y[t - 168 * H]


def test_rolling_windows_end_at_t_minus_1_and_exclude_y_t(raw, feats):
    y = _y(raw, "S002")
    f = feats[feats.series_id == "S002"].set_index("event_timestamp")
    for t in f.index[::41]:
        w24 = y[t - 24 * H : t - 1 * H]
        w168 = y[t - 168 * H : t - 1 * H]
        assert len(w24) == 24 and len(w168) == 168
        assert f.loc[t, "rolling_mean_24"] == pytest.approx(w24.mean())
        assert f.loc[t, "rolling_std_24"] == pytest.approx(w24.std(ddof=1))
        assert f.loc[t, "rolling_mean_168"] == pytest.approx(w168.mean())


def test_changing_y_at_k_never_changes_features_stamped_at_or_before_k(raw, feats):
    """THE leakage test: perturb one observation; no feature row stamped <= its timestamp may move."""
    k = raw[raw.series_id == "S001"]["event_timestamp"].iloc[400]
    perturbed = raw.copy()
    mask = (perturbed.series_id == "S001") & (perturbed.event_timestamp == k)
    perturbed.loc[mask, "target_value"] += 10_000.0
    f2 = build_features(perturbed)
    a = feats[(feats.series_id == "S001") & (feats.event_timestamp <= k)].reset_index(drop=True)
    b = f2[(f2.series_id == "S001") & (f2.event_timestamp <= k)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)  # untouched, including the row stamped k itself
    after = feats[(feats.series_id == "S001") & (feats.event_timestamp == k + H)]
    after2 = f2[(f2.series_id == "S001") & (f2.event_timestamp == k + H)]
    assert not np.isclose(after["value_lag_1"].iloc[0], after2["value_lag_1"].iloc[0])  # sanity: it does propagate forward


def test_truncating_the_future_does_not_change_past_rows(raw, feats):
    cut = raw.event_timestamp.min() + pd.Timedelta(hours=500)
    short = build_features(raw[raw.event_timestamp < cut], include_next_hour=False)
    full_past = feats[feats.event_timestamp < cut]
    merged = short.merge(full_past, on=["series_id", "event_timestamp"], suffixes=("_s", "_f"))
    assert len(merged) == len(short) > 0
    for c in FEATURE_NAMES:
        assert np.allclose(merged[f"{c}_s"], merged[f"{c}_f"])


def test_warmup_rows_not_emitted_and_next_hour_row_present(raw, feats):
    for sid, g in raw.groupby("series_id"):
        f = feats[feats.series_id == sid]
        assert f.event_timestamp.min() == g.event_timestamp.min() + 168 * H  # no imputation of warm-up
        assert f.event_timestamp.max() == g.event_timestamp.max() + 1 * H   # serving row, has no target


def test_gap_in_history_drops_dependent_rows_instead_of_filling(raw):
    g = raw[raw.series_id == "S003"]
    hole = g.event_timestamp.iloc[300]
    broken = raw[~((raw.series_id == "S003") & (raw.event_timestamp == hole))]
    f = build_features(broken)
    f3 = f[f.series_id == "S003"]
    # the row stamped at the gap only uses data BEFORE it, so it is legitimately computable
    # (its target is what is missing, so it can never become a training example)
    assert hole in set(f3.event_timestamp)
    assert not ((build_targets(broken).series_id == "S003") & (build_targets(broken).event_timestamp == hole)).any()
    assert not f3.event_timestamp.between(hole + H, hole + 168 * H).any()  # every window touching it
    assert (f3.event_timestamp > hole + 168 * H).any()                # recovers afterwards


def test_calendar_features_match_timestamp(feats):
    assert (feats.hour_of_day == feats.event_timestamp.dt.hour).all()
    assert (feats.day_of_week == feats.event_timestamp.dt.dayofweek).all()


def test_series_are_isolated(raw, feats):
    only = build_features(raw[raw.series_id == "S002"])
    pd.testing.assert_frame_equal(
        only.reset_index(drop=True),
        feats[feats.series_id == "S002"].reset_index(drop=True),
    )


def test_targets_are_value_at_t_and_feature_row_t_cannot_contain_it(raw, feats):
    tg = build_targets(raw)
    merged = feats.merge(tg, on=["series_id", "event_timestamp"])
    assert len(merged) > 0
    # target equals the observation stamped t, and is never equal-by-construction to any lag
    y = raw.set_index(["series_id", "event_timestamp"])["target_value"]
    sample = merged.iloc[::97]
    for _, r in sample.iterrows():
        assert r.target_value == y[(r.series_id, r.event_timestamp)]
    # the last feature row per series (serving row) has no target yet
    last = feats.groupby("series_id").event_timestamp.max()
    for sid, ts in last.items():
        assert not ((tg.series_id == sid) & (tg.event_timestamp == ts)).any()


def test_feature_set_id_is_stable():
    assert feature_set_id() == feature_set_id() and feature_set_id().startswith("fs-v1-")
