"""Feast against REAL Postgres (offline) and Redis (online). Skipped unless both are configured:
   TSP_TEST_DATABASE_URL=postgresql+psycopg2://user:pw@host:port/db   TSP_TEST_REDIS=host:port
"""
import os
import uuid

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from tsplatform.config.settings import DatabaseSettings, GeneratorConfig
from tsplatform.contracts.features import (
    ENTITY_KEY,
    FEATURE_NAMES,
    FEATURE_TTL,
    TIMESTAMP_COLUMN,
)
from tsplatform.data_access.db import create_app_schema, make_engine
from tsplatform.data_access.repositories import FeatureRepository, RawObservationRepository
from tsplatform.data_generation import compute_segments, generate_dataset
from tsplatform.feature_store import client
from tsplatform.feature_store.training_data import build_training_frame
from tsplatform.features import build_features, build_targets

PG = os.environ.get("TSP_TEST_DATABASE_URL")
REDIS = os.environ.get("TSP_TEST_REDIS")
pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (PG and REDIS), reason="TSP_TEST_DATABASE_URL / TSP_TEST_REDIS not set")]
H = pd.Timedelta(hours=1)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cfg = GeneratorConfig(seed=21, n_series=3, n_hours=24 * 7 * 6)
    engine = make_engine(DatabaseSettings(url_override=PG))
    create_app_schema(engine)
    with engine.begin() as c:  # isolation: this module owns the tables for its duration
        c.execute(text("DELETE FROM features.series_features"))
        c.execute(text("DELETE FROM raw.observations"))
    raw = generate_dataset(cfg)
    RawObservationRepository(engine).upsert(raw, "ds-test")
    feats = build_features(raw)
    FeatureRepository(engine).upsert(feats)

    url = make_url(PG)
    repo = tmp_path_factory.mktemp("feast_repo")
    project = f"tsp_test_{uuid.uuid4().hex[:8]}"
    (repo / "feature_store.yaml").write_text(f"""
project: {project}
provider: local
entity_key_serialization_version: 3
registry:
  registry_type: sql
  path: sqlite:///{repo}/registry.db
  cache_ttl_seconds: 0
offline_store:
  type: postgres
  host: {url.host}
  port: {url.port}
  database: {url.database}
  db_schema: features
  user: {url.username}
  password: "{url.password or ''}"
  sslmode: disable
online_store:
  type: redis
  connection_string: {REDIS}
""")
    store = client.get_store(repo)
    client.apply_definitions(store)
    yield {"cfg": cfg, "engine": engine, "raw": raw, "feats": feats, "store": store}
    try:
        store.teardown()
    except Exception:
        pass


def _expected(feats, sid, ts):
    r = feats[(feats.series_id == sid) & (feats.event_timestamp == ts)]
    return r.iloc[0][FEATURE_NAMES].astype(float)


def test_registry_has_definitions(world):
    s = world["store"]
    assert s.get_feature_service("forecast_features_v1") is not None
    assert [fv.name for fv in s.list_feature_views()] == ["series_features"]
    assert s.get_feature_view("series_features").ttl == FEATURE_TTL


def test_pit_returns_exactly_what_the_builder_computed(world):
    feats, store = world["feats"], world["store"]
    ent = build_targets(world["raw"])
    ent = ent[ent.event_timestamp >= feats.event_timestamp.min()].iloc[::13].reset_index(drop=True)
    got = client.historical_features(store, ent).sort_values([ENTITY_KEY, TIMESTAMP_COLUMN]).reset_index(drop=True)
    assert len(got) == len(ent)
    exp = ent.merge(feats, on=[ENTITY_KEY, TIMESTAMP_COLUMN]).sort_values([ENTITY_KEY, TIMESTAMP_COLUMN]).reset_index(drop=True)
    np.testing.assert_allclose(got[FEATURE_NAMES].to_numpy(float), exp[FEATURE_NAMES].to_numpy(float))


def test_pit_semantics_latest_row_at_or_before_t_never_after(world):
    feats, store = world["feats"], world["store"]
    sid = "S001"
    t = feats[feats.series_id == sid].event_timestamp.iloc[500]
    ent = pd.DataFrame({
        ENTITY_KEY: [sid] * 3,
        TIMESTAMP_COLUMN: [t, t + pd.Timedelta(minutes=30), t - pd.Timedelta(minutes=30)],
    })
    got = client.historical_features(store, ent).set_index(TIMESTAMP_COLUMN)
    np.testing.assert_allclose(got.loc[t, FEATURE_NAMES].to_numpy(float), _expected(feats, sid, t).to_numpy())
    np.testing.assert_allclose(got.loc[t + pd.Timedelta(minutes=30), FEATURE_NAMES].to_numpy(float),
                               _expected(feats, sid, t).to_numpy())           # t+30min -> row t, NOT t+1h
    np.testing.assert_allclose(got.loc[t - pd.Timedelta(minutes=30), FEATURE_NAMES].to_numpy(float),
                               _expected(feats, sid, t - H).to_numpy())       # t-30min -> row t-1h


def test_pit_before_first_feature_row_is_null_not_filled(world):
    feats, store = world["feats"], world["store"]
    first = feats[feats.series_id == "S002"].event_timestamp.min()
    ent = pd.DataFrame({ENTITY_KEY: ["S002"], TIMESTAMP_COLUMN: [first - H]})
    got = client.historical_features(store, ent)
    assert got[FEATURE_NAMES].isna().all(axis=None)


def test_ttl_makes_stale_features_unavailable(world):
    feats, store = world["feats"], world["store"]
    last = feats[feats.series_id == "S003"].event_timestamp.max()
    ent = pd.DataFrame({ENTITY_KEY: ["S003"] * 2,
                        TIMESTAMP_COLUMN: [last + FEATURE_TTL - H, last + FEATURE_TTL + H]})
    got = client.historical_features(store, ent).sort_values(TIMESTAMP_COLUMN)
    assert got.iloc[0][FEATURE_NAMES].notna().all()   # inside ttl: last row still valid
    assert got.iloc[1][FEATURE_NAMES].isna().all()    # beyond ttl: stale -> unavailable


def test_materialize_puts_only_rows_up_to_now_online_and_matches_training_view(world):
    feats, store = world["feats"], world["store"]
    now = feats[feats.series_id == "S001"].event_timestamp.iloc[800]
    client.materialize(store, end=now.to_pydatetime())  # first run: default start = end - TTL
    online = client.online_features(store, ["S001", "S002", "S003"]).set_index(ENTITY_KEY)
    assert (online.status == "ok").all()
    for sid in online.index:
        stamps = feats[(feats.series_id == sid) & (feats.event_timestamp <= now)].event_timestamp
        exp = _expected(feats, sid, stamps.max())
        np.testing.assert_allclose(online.loc[sid, FEATURE_NAMES].to_numpy(float), exp.to_numpy())
        # and it is not the future row
        nxt = feats[(feats.series_id == sid) & (feats.event_timestamp == stamps.max() + H)]
        assert not np.allclose(online.loc[sid, FEATURE_NAMES].to_numpy(float), nxt.iloc[0][FEATURE_NAMES].to_numpy(float))
    # training/serving consistency: online == PIT historical at the same instant
    ent = pd.DataFrame({ENTITY_KEY: ["S001"], TIMESTAMP_COLUMN: [now]})
    hist = client.historical_features(store, ent).iloc[0]
    np.testing.assert_allclose(online.loc["S001", FEATURE_NAMES].to_numpy(float), hist[FEATURE_NAMES].to_numpy(float))


def test_incremental_materialization_advances_online_state(world):
    feats, store = world["feats"], world["store"]
    t2 = feats[feats.series_id == "S001"].event_timestamp.iloc[830]
    client.materialize(store, end=t2.to_pydatetime())  # incremental from last interval
    online = client.online_features(store, ["S001"]).iloc[0]
    exp = _expected(feats, "S001", feats[(feats.series_id == "S001") & (feats.event_timestamp <= t2)].event_timestamp.max())
    np.testing.assert_allclose(online[FEATURE_NAMES].to_numpy(float), exp.to_numpy())


def test_materialize_never_moves_online_state_backwards(world):
    feats, store = world["feats"], world["store"]
    before = client.online_features(store, ["S001"]).iloc[0][FEATURE_NAMES].to_numpy(float)
    earlier = feats[feats.series_id == "S001"].event_timestamp.iloc[100]
    client.materialize(store, end=earlier.to_pydatetime())  # no-op: already materialised past it
    after = client.online_features(store, ["S001"]).iloc[0][FEATURE_NAMES].to_numpy(float)
    np.testing.assert_allclose(before, after)


def test_unknown_series_is_not_found_not_an_error(world):
    online = client.online_features(world["store"], ["S001", "NOPE"]).set_index(ENTITY_KEY)
    assert online.loc["NOPE", "status"] == "not_found"
    assert online.loc["NOPE", FEATURE_NAMES].isna().all()
    rep = client.verify_online_availability(world["store"], ["S001", "NOPE"])
    assert rep["missing"] == ["NOPE"] and rep["ok"] == 1


def test_training_frame_is_chronological_complete_and_has_lineage(world):
    cfg, store, engine = world["cfg"], world["store"], world["engine"]
    segs = compute_segments(cfg)
    raw_repo = RawObservationRepository(engine)
    train, ti = build_training_frame(store, raw_repo, segs["train"])
    val, vi = build_training_frame(store, raw_repo, segs["validation"])
    assert train.event_timestamp.max() < segs["train"].end <= val.event_timestamp.min()  # no overlap, ordered
    assert train[FEATURE_NAMES].notna().all(axis=None) and train.target_value.notna().all()
    assert ti.n_dropped_incomplete > 0 and vi.n_dropped_incomplete == 0   # warm-up only in the first segment
    assert ti.n_rows + ti.n_dropped_incomplete == ti.n_entity_rows
    assert ti.feature_set_id.startswith("fs-v1-") and len(ti.frame_hash) == 64
    # label alignment: target is the observation stamped t
    y = world["raw"].set_index([ENTITY_KEY, TIMESTAMP_COLUMN]).target_value
    for _, r in train.iloc[::211].iterrows():
        assert r.target_value == y[(r.series_id, r.event_timestamp)]
