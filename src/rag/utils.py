"""
Shared utilities for the rag package.

Imported before chromadb in every entry point — module-level code suppresses
ChromaDB telemetry noise at startup.
"""

import logging
import os
import sqlite3
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Suppress ChromaDB/posthog telemetry before chromadb is imported anywhere.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import posthog as _posthog
_posthog.capture = lambda *a, **kw: None  # chromadb 0.6.x / posthog 7.x signature mismatch

from .config import RagConfig, load_config


_LOG_CONFIGURED = False


def apply_offline_mode(config=None) -> bool:
    """Enforce fully-offline model loading when configured.

    When ``offline: true`` in config (or ``RAG_OFFLINE`` is set), export the
    Hugging Face offline flags so ``SentenceTransformer``/``CrossEncoder`` load
    only from the local cache and never reach the network — a cold cache then
    fails fast instead of silently downloading. Idempotent; returns whether
    offline mode is active. Explicit pre-set env values are left untouched.
    """
    offline = bool((config or {}).get("offline")) or bool(os.environ.get("RAG_OFFLINE"))
    if offline:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.setdefault(var, "1")
    return offline


class _SQLiteHandler(logging.Handler):
    """Logging handler that appends each record to a SQLite ``logs`` table.

    Columns: id, ts (local time), level, logger, message. logging serializes
    emit() with the handler lock, so a single shared connection is safe.

    ``retention_days`` prunes rows older than that many days on startup so the
    structured log cannot grow without bound on a long-lived server (0 disables).
    """

    def __init__(self, db_path: Path, retention_days: int = 30):
        super().__init__()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, level TEXT NOT NULL, "
            "logger TEXT NOT NULL, message TEXT NOT NULL)"
        )
        if retention_days and retention_days > 0:
            cutoff = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time() - retention_days * 86400),
            )
            self._conn.execute("DELETE FROM logs WHERE ts < ?", (cutoff,))
        self._conn.commit()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
            self._conn.execute(
                "INSERT INTO logs (ts, level, logger, message) VALUES (?, ?, ?, ?)",
                (ts, record.levelname, record.name, record.getMessage()),
            )
            self._conn.commit()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._conn.close()
        finally:
            super().close()


def setup_logging(config: dict | None = None, console: bool = True) -> logging.Logger:
    """Configure and return the shared ``rag`` logger.

    Attaches three handlers: a plain console handler (no timestamps, so the CLI
    looks unchanged), a rotating text file handler, and a SQLite handler that
    stores structured records. Paths are resolved as:
        text:    RAG_LOG_PATH    → config['log_path']    → ./logs/rag.log
        sqlite:  RAG_LOG_DB_PATH → config['log_db_path'] → <text path>.sqlite

    Pass ``console=False`` to skip the console handler (used by the query CLI so
    its results stay clean on stdout). Idempotent — safe to call more than once.
    Never raises on a bad path: it disables that handler and warns.
    """
    global _LOG_CONFIGURED
    logger = logging.getLogger("rag")
    if _LOG_CONFIGURED:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)

    cfg = config or {}
    log_path = Path(
        os.environ.get("RAG_LOG_PATH") or cfg.get("log_path") or "logs/rag.log"
    ).expanduser()

    # Rotating text log.
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning("File logging disabled (%s): %s", log_path, exc)

    # Structured SQLite log, alongside the text file.
    db_path = Path(
        os.environ.get("RAG_LOG_DB_PATH") or cfg.get("log_db_path")
        or log_path.with_suffix(".sqlite")
    ).expanduser()
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        retention_days = int(cfg.get("log_retention_days", 30))
        logger.addHandler(_SQLiteHandler(db_path, retention_days=retention_days))
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning("SQLite logging disabled (%s): %s", db_path, exc)

    _LOG_CONFIGURED = True
    logger.info("Logging to %s (+ %s)", log_path, db_path)
    return logger
