"""Unit tests for rag.drift — report-only source-vs-index drift detection.

Uses a fake embedding model and a real ChromaStore over a temp dir (the same
offline pattern as tests/test_indexing.py), so it runs with no GPU, no network,
and no real index. drift itself embeds nothing; the fake model is used only to
seed an in-sync index to compare against."""

import numpy as np
import pytest

from rag import drift
from rag.extractors import iter_sources
from rag.indexing import run_source
from rag.reconciliation import ReconciliationCatalog
from rag.store.chroma_store import ChromaStore
from pathlib import Path


class FakeModel:
    """Shape-correct, content-irrelevant stand-in for SentenceTransformer."""

    def encode(self, docs, normalize_embeddings=True, batch_size=16):
        return np.zeros((len(docs), 4))


def _write_note(vault: Path, name: str, body: str) -> None:
    (vault / "Knowledge").mkdir(parents=True, exist_ok=True)
    (vault / "Knowledge" / name).write_text(
        f"---\ntitle: {name}\ndomain: DevOps\n---\n# H\n{body}\n", encoding="utf-8"
    )


def _config(vault: Path, index_path: Path) -> dict:
    return {
        "vault_path": str(vault),
        "index_path": str(index_path),
        "collection_name": "drift",
        "exclude_dirs": [],
        "exclude_files": [],
        "markdown_workers": 1,
        "pdf_workers": 1,
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 150,
    }


def _seed_index(config: dict) -> ChromaStore:
    """Index the vault once with a fake model so the store matches the source."""
    vault = Path(config["vault_path"])
    index_path = Path(config["index_path"])
    store = ChromaStore(str(index_path), config["collection_name"])
    store.ensure(config["collection_name"])
    model = FakeModel()
    with ReconciliationCatalog.from_store(store, index_path) as catalog:
        for source in iter_sources(config, vault, 1200, 150):
            run_source(source, catalog, model, "cpu", 16, store)
    return store


def test_in_sync_reports_no_drift(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    sd = drift.check_source_drift(config)
    assert sd.state == "available"
    assert sd.missing_files == []
    assert sd.changed_files == []
    assert sd.stale_chunks == 0
    assert sd.drift is False


def test_added_file_reported_missing(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    # A note that exists in the source but was never indexed.
    _write_note(vault, "b.md", "fresh " * 300)
    sd = drift.check_source_drift(config)
    assert sd.missing_files == ["Knowledge/b.md"]
    assert sd.changed_files == []
    assert sd.drift is True


def test_changed_note_reported_and_leaves_stale(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    # Edit the body: chunks re-hash to new IDs, old ones become stale.
    _write_note(vault, "a.md", "totally different content " * 100)
    sd = drift.check_source_drift(config)
    assert sd.changed_files == ["Knowledge/a.md"]
    assert sd.stale_chunks > 0
    assert sd.drift is True


def test_non_git_vault_is_unknown_not_clean(monkeypatch):
    # Stub git so the test is independent of where the tmp dir lives: the
    # worktree is itself a git repo, so a real path-based check would otherwise
    # report in-tree. rev-parse --is-inside-work-tree returning non-"true" is
    # exactly what a non-git vault_path (e.g. a Jetson rsync target) produces.
    monkeypatch.setattr(drift, "_run_git", lambda args, cwd: None)
    git = drift.check_git_upstream(Path("/nonexistent/vault"))
    assert git.state == "unknown"
    assert git.is_git is False
    assert git.drift is False  # unknown never counts as drift
    assert "not a git work tree" in git.detail


def test_unknown_source_id_does_not_crash(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    sd = drift.check_source_drift(config, source_id="markdown:nope")
    assert sd.state == "unknown"
    assert sd.drift is False


def test_git_status_drift_logic():
    # behind > 0 => drift
    assert drift.GitStatus(state="ok", behind=3).drift is True
    # dirty > 0 => drift
    assert drift.GitStatus(state="ok", dirty=1).drift is True
    # ahead only, clean => no drift (host leads, but index can still match HEAD)
    assert drift.GitStatus(state="ok", ahead=5, behind=0, dirty=0).drift is False
    # in sync => no drift
    assert drift.GitStatus(state="ok", ahead=0, behind=0, dirty=0).drift is False
    # unknown => never drift
    assert drift.GitStatus(state="unknown", behind=99).drift is False


def test_main_exits_nonzero_on_drift(tmp_path, capsys, monkeypatch):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)
    _write_note(vault, "b.md", "fresh " * 300)  # missing from index -> drift

    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 1
    assert "DRIFT DETECTED" in capsys.readouterr().out


def test_main_exits_zero_when_in_sync(tmp_path, capsys, monkeypatch):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift", "--json"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 0
