"""Tests for the multi-corpus config profiles + pluggable store refactor
(docs/ADR-multi-corpus-profiles-and-pluggable-store.md).

Axis 1 (config profiles) is covered here now — profile-loading tests only,
offline, no chromadb/model involved. Axis 2 (RetrievalStore/ChromaStore
parity tests against a temp PersistentClient) lands later in this same file;
keep new sections clearly separated so both can coexist.
"""

from pathlib import Path

import pytest

from rag.utils import load_config

# tests/test_store.py -> repo root is one level up (see how src/rag/eval.py
# derives DEFAULT_GOLDEN with parents[2] from src/rag/eval.py; this file is
# one level shallower, at <repo_root>/tests/test_store.py).
REPO_ROOT = Path(__file__).resolve().parents[1]

MINILM = "sentence-transformers/all-MiniLM-L6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Axis 1 — config profiles
# ─────────────────────────────────────────────────────────────────────────────


def test_personal_profile_loads_expected_values(monkeypatch):
    monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / "config.personal.yaml"))

    cfg = load_config()

    assert cfg["collection_name"] == "obsidian_markdown"
    assert cfg["embedding_model"] == MINILM
    assert cfg["index_path"] == "./chroma_db"
    assert cfg["embedding_batch_size"] == 16
    assert cfg["store"] == "chroma"
    assert cfg["pdf_sources"]
    assert cfg["json_sources"]


def test_logmanager_profile_loads_expected_values(monkeypatch):
    monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / "config.logmanager.yaml"))

    cfg = load_config()

    assert cfg["collection_name"] == "wiki_lm"
    assert cfg["index_path"] == "./chroma_db_wiki"
    assert cfg["embedding_batch_size"] == 64
    assert cfg["markdown_workers"] > 1
    assert cfg["store"] == "chroma"

    # Markdown-only profile: no book/resource catalog pipeline.
    assert not cfg.get("pdf_sources")
    assert not cfg.get("json_sources")


def test_dev_default_config_uses_chroma_store(monkeypatch):
    monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / "config.yaml"))

    cfg = load_config()

    assert cfg["store"] == "chroma"
