import pandas as pd

from tsplatform.data_access.repositories import DatasetRegistry, RawObservationRepository
from tsplatform.data_generation import build_dataset_info, generate_dataset


def test_upsert_is_idempotent(sqlite_engine, small_cfg, monkeypatch):
    monkeypatch.setattr("tsplatform.data_access.repositories._MAX_PARAMS", 3000)  # force many chunks
    df = generate_dataset(small_cfg)
    info = build_dataset_info(small_cfg, df)
    repo = RawObservationRepository(sqlite_engine)
    repo.upsert(df, info.dataset_id)
    repo.upsert(df, info.dataset_id)  # retry / re-run
    assert repo.count() == len(df)


def test_roundtrip_preserves_data(sqlite_engine, small_cfg):
    df = generate_dataset(small_cfg)
    repo = RawObservationRepository(sqlite_engine)
    repo.upsert(df, "ds-x")
    back = repo.read()
    pd.testing.assert_frame_equal(
        df.reset_index(drop=True), back.reset_index(drop=True), check_dtype=False
    )


def test_upsert_updates_changed_values(sqlite_engine, small_cfg):
    df = generate_dataset(small_cfg)
    repo = RawObservationRepository(sqlite_engine)
    repo.upsert(df, "ds-1")
    changed = df.copy()
    changed.loc[0, "target_value"] = 999.0
    repo.upsert(changed, "ds-2")
    assert repo.count() == len(df)
    first = repo.read(series_ids=[df.loc[0, "series_id"]], end=df.loc[1, "event_timestamp"])
    assert first["target_value"].iloc[0] == 999.0


def test_read_filters_half_open_interval(sqlite_engine, small_cfg):
    df = generate_dataset(small_cfg)
    repo = RawObservationRepository(sqlite_engine)
    repo.upsert(df, "ds-1")
    start, end = pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-04")
    out = repo.read(series_ids=["S001"], start=start.to_pydatetime(), end=end.to_pydatetime())
    assert len(out) == 24
    assert out["event_timestamp"].min() == start and out["event_timestamp"].max() < end


def test_registry_register_and_get(sqlite_engine, small_cfg):
    df = generate_dataset(small_cfg)
    info = build_dataset_info(small_cfg, df)
    reg = DatasetRegistry(sqlite_engine)
    reg.register(info)
    reg.register(info)  # idempotent
    got = reg.get(info.dataset_id)
    assert got is not None and got.data_hash == info.data_hash and got.n_rows == len(df)
    assert reg.get("missing") is None
