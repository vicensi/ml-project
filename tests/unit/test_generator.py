import pandas as pd
import pytest

from tsplatform.config.settings import GeneratorConfig
from tsplatform.contracts.raw import validate_raw_frame
from tsplatform.data_generation import build_dataset_info, generate_dataset


def test_same_config_same_data(small_cfg):
    a, b = generate_dataset(small_cfg), generate_dataset(small_cfg)
    pd.testing.assert_frame_equal(a, b)
    assert build_dataset_info(small_cfg, a).dataset_id == build_dataset_info(small_cfg, b).dataset_id


def test_different_seed_different_data(small_cfg):
    other = small_cfg.model_copy(update={"seed": small_cfg.seed + 1})
    assert not generate_dataset(small_cfg)["target_value"].equals(generate_dataset(other)["target_value"])
    assert (build_dataset_info(small_cfg, generate_dataset(small_cfg)).dataset_id
            != build_dataset_info(other, generate_dataset(other)).dataset_id)


def test_adding_series_does_not_change_existing_ones(small_cfg):
    more = small_cfg.model_copy(update={"n_series": small_cfg.n_series + 2})
    a = generate_dataset(small_cfg)
    b = generate_dataset(more)
    b_sub = b[b["series_id"].isin(a["series_id"].unique())].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b_sub)


def test_schema_and_regular_hourly_frequency(small_cfg):
    df = generate_dataset(small_cfg)
    validate_raw_frame(df)
    assert df["series_id"].nunique() == small_cfg.n_series
    for _, g in df.groupby("series_id"):
        assert len(g) == small_cfg.n_hours
        assert (g["event_timestamp"].diff().dropna() == pd.Timedelta(hours=1)).all()
    assert (df["target_value"] >= 0).all()


def test_has_daily_seasonality_and_variety_between_series():
    cfg = GeneratorConfig(seed=1, n_series=4, n_hours=24 * 7 * 8, anomaly_rate=0.0)
    df = generate_dataset(cfg)
    for _, g in df.groupby("series_id"):
        by_hour = g.groupby(g["event_timestamp"].dt.hour)["target_value"].mean()
        assert by_hour.max() / by_hour.min() > 1.2  # clear daily cycle
    peaks = {sid: g.groupby(g["event_timestamp"].dt.hour)["target_value"].mean().idxmax()
             for sid, g in df.groupby("series_id")}
    assert len(set(peaks.values())) > 1  # different patterns per series


def test_anomalies_injected_only_when_enabled():
    clean = generate_dataset(GeneratorConfig(seed=3, n_series=2, n_hours=24 * 7 * 8, anomaly_rate=0.0))
    noisy = generate_dataset(GeneratorConfig(seed=3, n_series=2, n_hours=24 * 7 * 8, anomaly_rate=0.02))
    assert not clean["is_anomaly"].any()
    assert noisy["is_anomaly"].any()


def test_config_validation():
    with pytest.raises(ValueError):
        GeneratorConfig(n_hours=100)
    with pytest.raises(ValueError):
        GeneratorConfig(start=pd.Timestamp("2025-01-01 00:30").to_pydatetime())
