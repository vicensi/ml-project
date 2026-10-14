"""Command line entry points: ``python -m tsplatform.cli <command>``.

Commands
  ingest   generate synthetic data and persist it (idempotent)
  info     print the dataset identity and segment boundaries without touching the database
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from tsplatform.config.settings import GeneratorConfig, get_settings
from tsplatform.data_access.db import create_app_schema, make_engine
from tsplatform.data_access.repositories import DatasetRegistry, RawObservationRepository
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
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.service_name, settings.log_level, stream=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
