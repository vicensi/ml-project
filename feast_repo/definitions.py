"""Entry point for the `feast` CLI (feast apply / feast materialize ...).

Definitions live in the package so that training, serving and tests import the same objects.
"""
from tsplatform.feature_store.definitions import (  # noqa: F401
    forecast_feature_service,
    series,
    series_features_source,
    series_features_view,
)
