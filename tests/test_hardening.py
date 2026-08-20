"""Offline tests for Task 11 cross-cutting hardening.

Covers offline-mode env enforcement, SQLite log retention, JWT secret-strength
validation, and query-text log redaction. No network, no models.
"""

import sqlite3
import time

import pytest

import rag.query as q
from rag.api.auth import MIN_SECRET_BYTES, validate_secret_strength
from rag.utils import _SQLiteHandler, apply_offline_mode


# ── offline mode ─────────────────────────────────────────────────────────────


def test_apply_offline_mode_sets_hf_flags(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("RAG_OFFLINE", raising=False)
    assert apply_offline_mode({"offline": True}) is True
    import os
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_apply_offline_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("RAG_OFFLINE", raising=False)
    assert apply_offline_mode({}) is False


def test_apply_offline_mode_env_trigger(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("RAG_OFFLINE", "1")
    assert apply_offline_mode({}) is True


# ── SQLite log retention ─────────────────────────────────────────────────────


def test_sqlite_handler_prunes_old_rows(tmp_path):
    db = tmp_path / "log.sqlite"
    # Seed a row far in the past and one now, then attach a 30-day-retention handler.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE logs (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, "
        "logger TEXT, message TEXT)"
    )
    old_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 90 * 86400))
    new_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    conn.execute("INSERT INTO logs (ts, level, logger, message) VALUES (?,?,?,?)",
                 (old_ts, "INFO", "rag", "ancient"))
    conn.execute("INSERT INTO logs (ts, level, logger, message) VALUES (?,?,?,?)",
                 (new_ts, "INFO", "rag", "recent"))
    conn.commit()
    conn.close()

    handler = _SQLiteHandler(db, retention_days=30)
    handler.close()

    conn = sqlite3.connect(str(db))
    messages = {row[0] for row in conn.execute("SELECT message FROM logs")}
    conn.close()
    assert messages == {"recent"}


def test_sqlite_handler_retention_zero_keeps_all(tmp_path):
    db = tmp_path / "log.sqlite"
    handler = _SQLiteHandler(db, retention_days=0)
    handler.close()  # table created, nothing pruned
    conn = sqlite3.connect(str(db))
    # No error and table exists.
    conn.execute("SELECT COUNT(*) FROM logs")
    conn.close()


# ── JWT secret strength ──────────────────────────────────────────────────────


def test_validate_secret_strength_accepts_strong_and_unset():
    validate_secret_strength(None)  # unset is deferred to per-request 500
    validate_secret_strength("x" * MIN_SECRET_BYTES)


def test_validate_secret_strength_rejects_weak():
    with pytest.raises(RuntimeError, match="too short"):
        validate_secret_strength("short")


# ── query-text log redaction ─────────────────────────────────────────────────


class _FakeModel:
    def encode(self, texts, **kw):
        import numpy as np
        return np.zeros((1, 8), dtype="float32")


class _FakeStore:
    def query(self, embedding, k, where=None, *, text=None, hybrid=False):
        return [{"document": "d", "metadata": {"title": "d"}, "distance": 0.1}]


def test_query_text_redacted_by_default(caplog):
    with caplog.at_level("INFO", logger="rag"):
        q.search("super secret question", n_results=1,
                 config={"embedding_model": "x"}, model=_FakeModel(), store=_FakeStore())
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "super secret question" not in joined
    assert "redacted" in joined


def test_query_text_logged_when_opted_in(caplog):
    with caplog.at_level("INFO", logger="rag"):
        q.search("visible question", n_results=1,
                 config={"embedding_model": "x", "log_queries": True},
                 model=_FakeModel(), store=_FakeStore())
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "visible question" in joined


# ── inference concurrency limiter ────────────────────────────────────────────


def test_inference_limiter_rejects_when_full_and_recovers():
    from fastapi import HTTPException

    from rag.api.concurrency import InferenceLimiter, guard

    limiter = InferenceLimiter(max_concurrency=1, acquire_timeout=0.05)
    with limiter.slot():
        # No free slot within the timeout → 503.
        with pytest.raises(HTTPException) as excinfo:
            with limiter.slot():
                pass
        assert excinfo.value.status_code == 503
    # Slot released → acquirable again.
    with limiter.slot():
        pass


def test_guard_is_noop_without_limiter():
    from rag.api.concurrency import guard

    with guard({}):  # no inference_limiter key → no-op
        pass
    with guard({"inference_limiter": None}):
        pass
