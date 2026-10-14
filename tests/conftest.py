import pytest

from tsplatform.config.settings import DatabaseSettings, GeneratorConfig
from tsplatform.data_access.db import create_app_schema, make_engine


@pytest.fixture
def small_cfg() -> GeneratorConfig:
    return GeneratorConfig(seed=7, n_series=3, n_hours=24 * 7 * 6)


@pytest.fixture
def sqlite_engine():
    # in-memory sqlite: keep a single connection so all sessions see the same DB
    from sqlalchemy.pool import StaticPool

    engine = make_engine(
        DatabaseSettings(url_override="sqlite://"), poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    create_app_schema(engine)
    return engine
