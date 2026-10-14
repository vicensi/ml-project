import io
import json
import logging

import pytest

from tsplatform.config.settings import DatabaseSettings, Settings
from tsplatform.observability import bind_context, clear_context, configure_logging, log_operation


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("TSP_GEN_SEED", "123")
    monkeypatch.setenv("TSP_GEN_N_SERIES", "9")
    monkeypatch.setenv("TSP_DB_PASSWORD", "s3cret")
    s = Settings(_env_file=None)
    assert s.generator.seed == 123 and s.generator.n_series == 9
    assert "s3cret" not in repr(s) and "s3cret" not in str(s.database)


def test_database_url_never_hardcodes_credentials():
    url = DatabaseSettings(host="h", user="u", password="p", name="n").sqlalchemy_url()
    assert url.drivername == "postgresql+psycopg2" and url.host == "h"


def test_json_logs_have_required_fields_and_context():
    buf = io.StringIO()
    configure_logging("svc", "INFO", stream=buf)
    log = logging.getLogger("t")
    bind_context(request_id="req-1")
    with log_operation(log, "demo", model_version="3"):
        pass
    clear_context()
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    for key in ("timestamp", "level", "service", "operation", "request_id", "duration_ms", "outcome", "model_version"):
        assert key in rec
    assert rec["outcome"] == "ok" and rec["request_id"] == "req-1"


def test_log_operation_categorises_errors():
    buf = io.StringIO()
    configure_logging("svc", "INFO", stream=buf)

    class Down(Exception):
        error_category = "infrastructure"

    with pytest.raises(Down):
        with log_operation(logging.getLogger("t"), "call_feast"):
            raise Down()
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rec["outcome"] == "error" and rec["error_category"] == "infrastructure"
