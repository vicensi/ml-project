"""JSON logging on top of the stdlib.

Every record carries: timestamp (UTC ISO-8601), level, service, logger, message and any
context bound via ``bind_context`` (request_id, run_id, model_version...). ``log_operation``
adds operation name, duration, outcome and a categorised error — the fields required for
tracing a prediction back to its request/model/run.

Error categories: ``business`` (caller's fault / expected), ``infrastructure`` (dependency
down), ``unexpected``. Callers pass the category by raising exceptions that define
``error_category``; anything else is ``unexpected``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})
_RESERVED = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context.get())
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            # type only — stack traces stay in logs, never in HTTP responses
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(payload, default=str)


def configure_logging(service: str, level: str = "INFO", stream: Any = None) -> None:
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def bind_context(**fields: Any) -> None:
    _context.set({**_context.get(), **fields})


def clear_context() -> None:
    _context.set({})


@contextlib.contextmanager
def log_operation(logger: logging.Logger, operation: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Log start-less, single-line outcome with duration. Yields a dict to add result fields."""
    extra: dict[str, Any] = dict(fields)
    start = time.perf_counter()
    try:
        yield extra
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        category = getattr(exc, "error_category", "unexpected")
        logger.error(
            f"{operation} failed",
            extra={**extra, "operation": operation, "duration_ms": duration_ms,
                   "outcome": "error", "error_category": category,
                   "error_type": type(exc).__name__},
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            f"{operation} ok",
            extra={**extra, "operation": operation, "duration_ms": duration_ms, "outcome": "ok"},
        )
