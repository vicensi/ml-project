"""Synthetic demand generator.

Design decisions
* Determinism: each series draws from its own stream
  ``SeedSequence(entropy=seed, spawn_key=(series_index, purpose))``. Consequence: series i is identical
  whether you generate 3 or 300 series with the same seed (adding series never changes old ones).
* Components: level + linear trend + daily seasonality (two harmonics) + weekly seasonality
  + heteroscedastic noise + short multiplicative anomaly events.
* Patterns differ per series (office / residential / industrial / retail) so the entity key
  carries real information.
* Output is clipped at 0 (demand cannot be negative). Zeros are rare but possible, which is why
  evaluation uses WAPE rather than plain MAPE.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from tsplatform.config.settings import GeneratorConfig
from tsplatform.contracts.raw import DatasetInfo, validate_raw_frame

Pattern = Literal["office", "residential", "industrial", "retail"]
_PATTERNS: tuple[Pattern, ...] = ("office", "residential", "industrial", "retail")


@dataclass(frozen=True)
class SeriesProfile:
    series_id: str
    pattern: Pattern
    level: float
    trend_per_hour: float
    daily_amp: float
    daily_peak_hour: float
    harmonic2_amp: float
    weekly_amp: float  # relative weekday/weekend swing (sign depends on pattern)
    noise_rel_std: float


def series_id_for(index: int) -> str:
    return f"S{index + 1:03d}"


def _stream(seed: int, index: int, purpose: int) -> np.random.Generator:
    """purpose 0 = profile parameters, 1 = signal noise/anomalies (independent streams)."""
    return np.random.default_rng(np.random.SeedSequence(entropy=seed, spawn_key=(index, purpose)))


def make_profile(seed: int, index: int) -> SeriesProfile:
    rng = _stream(seed, index, 0)
    pattern = _PATTERNS[index % len(_PATTERNS)]  # deterministic mix, guarantees variety
    level = float(rng.uniform(80, 220))
    peak = {"office": 14.0, "residential": 20.0, "industrial": 11.0, "retail": 17.0}[pattern]
    weekly = {"office": -0.30, "residential": 0.12, "industrial": -0.05, "retail": 0.25}[pattern]
    return SeriesProfile(
        series_id=series_id_for(index),
        pattern=pattern,
        level=level,
        trend_per_hour=float(rng.uniform(-0.002, 0.008)) * level / 100,
        daily_amp=float(rng.uniform(0.20, 0.45)) * level,
        daily_peak_hour=peak + float(rng.normal(0, 0.7)),
        harmonic2_amp=float(rng.uniform(0.03, 0.12)) * level,
        weekly_amp=weekly * level * float(rng.uniform(0.7, 1.3)),
        noise_rel_std=float(rng.uniform(0.02, 0.06)),
    )


def _generate_one(cfg: GeneratorConfig, index: int, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    p = make_profile(cfg.seed, index)
    rng = _stream(cfg.seed, index, 1)

    n = len(timestamps)
    t = np.arange(n, dtype=float)
    hour = timestamps.hour.to_numpy(dtype=float)
    dow = timestamps.dayofweek.to_numpy()

    daily = p.daily_amp * np.cos(2 * np.pi * (hour - p.daily_peak_hour) / 24.0)
    daily2 = p.harmonic2_amp * np.cos(4 * np.pi * (hour - p.daily_peak_hour) / 24.0)
    weekend = (dow >= 5).astype(float)
    weekly = p.weekly_amp * weekend  # weekend shift relative to weekdays
    signal = p.level + p.trend_per_hour * t + daily + daily2 + weekly

    noise = rng.normal(0.0, 1.0, n) * p.noise_rel_std * np.maximum(signal, 1.0)
    value = signal + noise

    # anomaly events: Bernoulli start per hour, duration 1-6h, +-(30..120)% multiplicative
    is_anomaly = np.zeros(n, dtype=bool)
    starts = np.flatnonzero(rng.random(n) < cfg.anomaly_rate)
    for s in starts:
        dur = int(rng.integers(1, 7))
        mag = float(rng.uniform(0.3, 1.2)) * (1.0 if rng.random() < 0.7 else -1.0)
        sl = slice(s, min(s + dur, n))
        value[sl] = value[sl] * (1.0 + mag)
        is_anomaly[sl] = True

    value = np.round(np.clip(value, 0.0, None), 4)
    return pd.DataFrame(
        {
            "series_id": p.series_id,
            "event_timestamp": timestamps,
            "target_value": value,
            "is_anomaly": is_anomaly,
        }
    )


def generate_dataset(cfg: GeneratorConfig) -> pd.DataFrame:
    """Return the full raw history, sorted by (series_id, event_timestamp)."""
    timestamps = pd.date_range(cfg.start, periods=cfg.n_hours, freq="h")
    frames = [_generate_one(cfg, i, timestamps) for i in range(cfg.n_series)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["series_id", "event_timestamp"], kind="stable").reset_index(drop=True)
    validate_raw_frame(df)
    return df


def _config_json(cfg: GeneratorConfig) -> str:
    return json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def build_dataset_info(cfg: GeneratorConfig, df: pd.DataFrame) -> DatasetInfo:
    """Content-addressed identity: config hash + data hash. Logged to MLflow in Phase 3."""
    cfg_json = _config_json(cfg)
    config_hash = hashlib.sha256(cfg_json.encode()).hexdigest()
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    data_hash = hashlib.sha256(row_hashes.tobytes()).hexdigest()
    return DatasetInfo(
        dataset_id=f"ds-{config_hash[:8]}-{data_hash[:8]}",
        config_hash=config_hash,
        data_hash=data_hash,
        config_json=cfg_json,
        n_rows=len(df),
        n_series=df["series_id"].nunique(),
        ts_min=df["event_timestamp"].min().to_pydatetime(),
        ts_max=df["event_timestamp"].max().to_pydatetime(),
    )
