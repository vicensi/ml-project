"""Command line entry points: ``python -m tsplatform.cli <command>``.

Commands
  ingest            generate synthetic data and persist it (idempotent)
  info              print the dataset identity and segment boundaries (no database)
  build-features    raw history -> leak-free features -> offline table (idempotent)
  feast-apply       register entity / feature view / feature service in the Feast registry
  feast-materialize copy the latest features (stamp <= --end) to the online store (Redis)
  verify-features   every series must have complete online features (exit code 1 otherwise)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from tsplatform.config.settings import GeneratorConfig, get_settings
from tsplatform.data_access.db import create_app_schema, make_engine
from tsplatform.data_access.repositories import (
    DatasetRegistry,
    FeatureRepository,
    RawObservationRepository,
)
from tsplatform.data_generation.generator import build_dataset_info, generate_dataset
from tsplatform.data_generation.segments import compute_segments
from tsplatform.observability.logging import configure_logging, log_operation

log = logging.getLogger("tsplatform.cli")


def _gen_config(args: argparse.Namespace) -> GeneratorConfig:
    base = get_settings().generator
    overrides = {
        k: v
        for k, v in {"seed": args.seed, "n_series": args.n_series, "n_hours": args.n_hours}.items()
        if v is not None
    }
    return GeneratorConfig(**{**base.model_dump(), **overrides})


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    cfg = _gen_config(args)
    with log_operation(log, "generate_synthetic_data", seed=cfg.seed, n_series=cfg.n_series) as r:
        df = generate_dataset(cfg)
        info = build_dataset_info(cfg, df)
        r["dataset_id"] = info.dataset_id
        r["rows"] = len(df)
    engine = make_engine(settings.database)
    with log_operation(log, "persist_raw_data", dataset_id=info.dataset_id) as r:
        create_app_schema(engine)
        DatasetRegistry(engine).register(info)
        r["rows_written"] = RawObservationRepository(engine).upsert(df, info.dataset_id)
    return 0


def cmd_build_features(args: argparse.Namespace) -> int:
    from tsplatform.features import build_features  # lazy: keeps `ingest` import-light

    engine = make_engine(get_settings().database)
    create_app_schema(engine)
    with log_operation(log, "build_historical_features") as r:
        raw = RawObservationRepository(engine).read()
        feats = build_features(raw)
        r["raw_rows"], r["feature_rows"] = len(raw), len(feats)
        r["rows_written"] = FeatureRepository(engine).upsert(feats)
    return 0


def _store():
    from tsplatform.feature_store.client import get_store

    return get_store(get_settings().feast_repo_path)


def cmd_feast_apply(args: argparse.Namespace) -> int:
    from tsplatform.feature_store.client import apply_definitions

    apply_definitions(_store())
    return 0


def cmd_feast_materialize(args: argparse.Namespace) -> int:
    from tsplatform.feature_store.client import materialize

    end = args.end
    if end is None:  # default "now" = the serving row right after the last observed hour
        end = FeatureRepository(make_engine(get_settings().database)).max_timestamp()
        if end is None:
            log.error("no features in the offline table; run build-features first")
            return 1
    materialize(_store(), end=end, start=args.start)
    return 0


def cmd_verify_features(args: argparse.Namespace) -> int:
    from tsplatform.feature_store.client import verify_online_availability

    engine = make_engine(get_settings().database)
    report = verify_online_availability(_store(), RawObservationRepository(engine).series_ids())
    print(json.dumps(report))
    log.info("verify_feature_availability", extra=report)
    return 0 if not report["missing"] else 1


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _gen_config(args)
    df = generate_dataset(cfg)
    info = build_dataset_info(cfg, df)
    segs = {k: [v.start.isoformat(), v.end.isoformat()] for k, v in compute_segments(cfg).items()}
    print(json.dumps({"dataset": info.model_dump(mode="json"), "segments": segs}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tsplatform")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("ingest", cmd_ingest), ("info", cmd_info)):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int)
        p.add_argument("--n-series", type=int)
        p.add_argument("--n-hours", type=int)
        p.set_defaults(func=fn)
    sub.add_parser("build-features").set_defaults(func=cmd_build_features)
    sub.add_parser("feast-apply").set_defaults(func=cmd_feast_apply)
    sub.add_parser("verify-features").set_defaults(func=cmd_verify_features)
    pm = sub.add_parser("feast-materialize")
    pm.add_argument("--end", type=datetime.fromisoformat, help="ISO timestamp, naive UTC (default: latest feature stamp)")
    pm.add_argument("--start", type=datetime.fromisoformat, help="omit for incremental materialisation")
    pm.set_defaults(func=cmd_feast_materialize)
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.service_name, settings.log_level, stream=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
