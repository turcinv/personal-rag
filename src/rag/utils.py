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

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _find_config() -> Path:
    """Locate config.yaml: RAG_CONFIG_PATH env var → cwd → package root."""
    if cfg_env := os.environ.get("RAG_CONFIG_PATH"):
        return Path(cfg_env)
    cwd_cfg = Path.cwd() / "config.yaml"
    if cwd_cfg.exists():
        return cwd_cfg
    # Editable install: src/rag/utils.py → ../../../ = project root
    return Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config() -> dict:
    """Load config.yaml and apply environment variable overrides.

    RAG indexer overrides:
        RAG_VAULT_PATH          overrides vault_path
        RAG_PDF_BOOKS_PATH      overrides pdf_sources entry with type=book
        RAG_PDF_RESOURCES_PATH  overrides pdf_sources entry with type=resource
        RAG_JSON_PATH           overrides json_sources to a single directory
        RAG_INDEX_PATH          overrides index_path (ChromaDB storage dir)
        RAG_CONFIG_PATH         overrides the config.yaml location itself

    Extractor pipeline overrides (config.extractor.*):
        RAG_BOOKS_PATH          overrides extractor.books_path
        RAG_RESOURCES_PATH      overrides extractor.resources_path
        RAG_CATALOG_PATH        overrides extractor.catalog_path
        RAG_OUTPUT_PATH         overrides extractor.output_path
        RAG_OBSIDIAN_NOTES_PATH overrides extractor.obsidian_notes_path
        RAG_MOCS_PATH           overrides extractor.mocs_path
    """
    config_path = _find_config()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # RAG indexer overrides.
    if vault := os.environ.get("RAG_VAULT_PATH"):
        cfg["vault_path"] = vault
    if books := os.environ.get("RAG_PDF_BOOKS_PATH"):
        for src in cfg.get("pdf_sources", []):
            if src.get("type") == "book":
                src["path"] = books
    if resources := os.environ.get("RAG_PDF_RESOURCES_PATH"):
        for src in cfg.get("pdf_sources", []):
            if src.get("type") == "resource":
                src["path"] = resources
    if json_path := os.environ.get("RAG_JSON_PATH"):
        cfg["json_sources"] = [{"path": json_path}]
    if index := os.environ.get("RAG_INDEX_PATH"):
        cfg["index_path"] = index

    # Extractor pipeline overrides.
    ext = cfg.setdefault("extractor", {})
    if v := os.environ.get("RAG_BOOKS_PATH"):
        ext["books_path"] = v
    if v := os.environ.get("RAG_RESOURCES_PATH"):
        ext["resources_path"] = v
    if v := os.environ.get("RAG_CATALOG_PATH"):
        ext["catalog_path"] = v
    if v := os.environ.get("RAG_OUTPUT_PATH"):
        ext["output_path"] = v
    if v := os.environ.get("RAG_OBSIDIAN_NOTES_PATH"):
        ext["obsidian_notes_path"] = v
    if v := os.environ.get("RAG_MOCS_PATH"):
        ext["mocs_path"] = v

    return cfg


_LOG_CONFIGURED = False


class _SQLiteHandler(logging.Handler):
    """Logging handler that appends each record to a SQLite ``logs`` table.

    Columns: id, ts (local time), level, logger, message. logging serializes
    emit() with the handler lock, so a single shared connection is safe.
    """

    def __init__(self, db_path: Path):
        super().__init__()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS logs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, level TEXT NOT NULL, "
            "logger TEXT NOT NULL, message TEXT NOT NULL)"
        )
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
        logger.addHandler(_SQLiteHandler(db_path))
    except (OSError, sqlite3.Error) as exc:
        logger.warning("SQLite logging disabled (%s): %s", db_path, exc)

    _LOG_CONFIGURED = True
    logger.info("Logging to %s (+ %s)", log_path, db_path)
    return logger
