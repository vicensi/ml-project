"""Typed settings.

Layers (later wins): defaults < env file (TSP_ENV_FILE, default ".env") < process environment.
Local vs production differ only by which env file / secret manager feeds the process;
code never branches on secrets.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL


class SegmentConfig(BaseModel):
    """Chronological split of the synthetic timeline (fractions of total hours).

    train -> fit models; validation -> model selection; test -> final offline estimate;
    production -> held back and "released" progressively to simulate live traffic.
    """

    train: float = 0.60
    validation: float = 0.15
    test: float = 0.10
    production: float = 0.15

    @model_validator(mode="after")
    def _sum_to_one(self) -> SegmentConfig:
        total = self.train + self.validation + self.test + self.production
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"segment fractions must sum to 1.0, got {total}")
        if min(self.train, self.validation, self.test, self.production) <= 0:
            raise ValueError("every segment fraction must be > 0")
        return self


class GeneratorConfig(BaseModel):
    """Everything that determines the synthetic dataset. Same config => identical data."""

    seed: int = 42
    n_series: int = Field(5, ge=1, le=1000)
    start: datetime = datetime(2025, 1, 1)  # naive, interpreted as UTC
    n_hours: int = Field(24 * 7 * 26, description="Hourly points per series")
    anomaly_rate: float = Field(0.002, ge=0.0, le=0.05, description="P(anomaly starts) per hour")
    segments: SegmentConfig = SegmentConfig()

    @field_validator("start")
    @classmethod
    def _naive_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is not None:
            raise ValueError("start must be timezone-naive and expressed in UTC")
        if v.minute or v.second or v.microsecond:
            raise ValueError("start must be aligned to a whole hour")
        return v

    @field_validator("n_hours")
    @classmethod
    def _enough_history(cls, v: int) -> int:
        # lag_168 + rolling_mean_168 need one week of warm-up, plus room for every segment.
        if v < 24 * 7 * 4:
            raise ValueError("n_hours must cover at least 4 weeks (672h)")
        return v


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "tsplatform"
    user: str = "tsp_app"
    password: SecretStr = SecretStr("")
    # Full override (e.g. sqlite:///:memory: in unit tests). Takes precedence when set.
    url_override: str | None = None

    def sqlalchemy_url(self) -> str | URL:
        if self.url_override:
            return self.url_override
        return URL.create(
            "postgresql+psycopg2",
            username=self.user,
            password=self.password.get_secret_value(),
            host=self.host,
            port=self.port,
            database=self.name,
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TSP_",
        env_file=os.environ.get("TSP_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "test", "production"] = "local"
    service_name: str = "tsplatform"
    log_level: str = "INFO"

    # Flat env names (TSP_DB_HOST ...) mapped explicitly: nested delimiters are error-prone.
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "tsplatform"
    db_user: str = "tsp_app"
    db_password: SecretStr = SecretStr("")
    db_url: str | None = None

    gen_seed: int = 42
    gen_n_series: int = 5
    gen_n_hours: int = 24 * 7 * 26
    gen_start: datetime = datetime(2025, 1, 1)
    gen_anomaly_rate: float = 0.002

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings(
            host=self.db_host,
            port=self.db_port,
            name=self.db_name,
            user=self.db_user,
            password=self.db_password,
            url_override=self.db_url,
        )

    @property
    def generator(self) -> GeneratorConfig:
        return GeneratorConfig(
            seed=self.gen_seed,
            n_series=self.gen_n_series,
            n_hours=self.gen_n_hours,
            start=self.gen_start,
            anomaly_rate=self.gen_anomaly_rate,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
