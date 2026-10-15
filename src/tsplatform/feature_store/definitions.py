"""Feast object definitions (single source of truth, imported by feast_repo/definitions.py).

  Entity         series            join key series_id
  Source         features.series_features (PostgreSQL offline store)
  FeatureView    series_features   9 features, ttl = FEATURE_TTL, online=True
  FeatureService forecast_features_v1   <- the ONLY thing training and serving ask for
"""

from __future__ import annotations

from feast import Entity, FeatureService, FeatureView, Field, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int32

from tsplatform.contracts.features import (
    ENTITY_KEY,
    ENTITY_NAME,
    FEATURE_SERVICE_NAME,
    FEATURE_TTL,
    FEATURE_VIEW_NAME,
    FLOAT_FEATURES,
    INT_FEATURES,
    TIMESTAMP_COLUMN,
    feature_set_id,
)

series = Entity(
    name=ENTITY_NAME,
    join_keys=[ENTITY_KEY],
    value_type=ValueType.STRING,
    description="One hourly demand/consumption time series",
)

series_features_source = PostgreSQLSource(
    name="series_features_source",
    query="SELECT * FROM features.series_features",
    timestamp_field=TIMESTAMP_COLUMN,
)

series_features_view = FeatureView(
    name=FEATURE_VIEW_NAME,
    entities=[series],
    ttl=FEATURE_TTL,
    schema=[Field(name=n, dtype=Float64) for n in FLOAT_FEATURES]
    + [Field(name=n, dtype=Int32) for n in INT_FEATURES],
    source=series_features_source,
    online=True,
    tags={"feature_set_id": feature_set_id()},
    description="Lags, rolling statistics and calendar features; row t uses only data < t",
)

forecast_feature_service = FeatureService(
    name=FEATURE_SERVICE_NAME,
    features=[series_features_view],
    tags={"feature_set_id": feature_set_id()},
)

ALL_OBJECTS = [series, series_features_source, series_features_view, forecast_feature_service]
