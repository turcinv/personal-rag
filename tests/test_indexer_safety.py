"""Integration tests for source-scoped index planning and deletion safety."""

from pathlib import Path

import numpy as np
import pytest

import rag.indexer as indexer
from rag.extractors import Source


class FakeModel:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, docs, normalize_embeddings=True, batch_size=16):
        return np.zeros((len(docs), 4))


class FakeStore:
    name = "test"

    def __init__(self, existing):
        self.existing = existing
        self.deleted = []
        self.upserts = []
        self.updates = []
        self.ensure_calls = []

    def ensure(self, name):
        self.ensure_calls.append(name)

    def count(self):
        return len(self.existing)

    def iter_metadata(self, page_size=5_000):
        del page_size
        yield from self.existing.items()

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def update_metadata(self, ids, metas):
        self.updates.append((ids, metas))

    def delete(self, ids):
        self.deleted.extend(ids)


def _source(tmp_path, source_id, state="available", files=True):
    path = tmp_path / f"{source_id.replace(':', '-')}.md"
    if files:
        path.write_text("body")
    return Source(
        source_id=source_id,
        kind="markdown" if source_id.startswith("markdown") else "json",
        label=source_id,
        root=tmp_path,
        files=[path] if files else [],
        workers=1,
        extract=lambda _path: (
            ["healthy-current"],
            ["body"],
            [{"path": "current.md"}],
            None,
        ),
        file_key=lambda item: item.name,
        state=state,
        detail="mount missing" if state == "missing" else "",
    )


def _config(tmp_path):
    return {
        "vault_path": str(tmp_path),
        "collection_name": "test",
        "index_path": str(tmp_path / "chroma"),
        "embedding_model": "fake",
        "embedding_batch_size": 16,
        "markdown_workers": 1,
        "pdf_workers": 1,
        "prune_max_fraction": 1.0,
        "prune_max_chunks": 100,
        "manifest_path": str(tmp_path / "manifests"),
    }


def test_partial_missing_source_is_preserved_while_healthy_source_updates(tmp_path, monkeypatch):
    healthy = _source(tmp_path, "markdown:vault")
    missing = _source(tmp_path, "json:catalog", state="missing", files=False)
    existing = {
        "healthy-current": {
            "path": "current.md",
            "source_id": "markdown:vault",
            "source_kind": "markdown",
            "file_id": healthy.files[0].name,
        },
        "healthy-deleted": {
            "path": "deleted.md",
            "source_id": "markdown:vault",
            "source_kind": "markdown",
            "file_id": "deleted.md",
        },
        "missing-old": {
            "path": "book.pdf",
            "source_id": "json:catalog",
            "source_kind": "json",
            "file_id": "book.json",
        },
    }
    store = FakeStore(existing)
    monkeypatch.setattr(indexer, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(indexer, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(indexer, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(indexer, "get_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(indexer, "iter_sources", lambda *_args, **_kwargs: [healthy, missing])
    monkeypatch.setattr("sys.argv", ["rag-index"])

    indexer.main()

    assert store.deleted == ["healthy-deleted"]
    assert "missing-old" not in store.deleted
    manifests = list((tmp_path / "manifests" / "test").glob("*.json"))
    assert len(manifests) == 1


def test_dry_run_performs_no_store_writes_or_manifest_write(tmp_path, monkeypatch):
    healthy = _source(tmp_path, "markdown:vault")
    store = FakeStore({})
    monkeypatch.setattr(indexer, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(indexer, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(indexer, "get_store", lambda *_args, **_kwargs: store)
    monkeypatch.setattr(indexer, "iter_sources", lambda *_args, **_kwargs: [healthy])
    monkeypatch.setattr("sys.argv", ["rag-index", "--dry-run"])

    indexer.main()

    assert store.ensure_calls == []
    assert store.upserts == []
    assert store.updates == []
    assert store.deleted == []
    assert not (tmp_path / "manifests").exists()


def test_duplicate_source_ids_fail_before_model_or_store_writes(tmp_path, monkeypatch):
    first = _source(tmp_path, "json:duplicate")
    second = _source(tmp_path, "json:duplicate")
    monkeypatch.setattr(indexer, "load_config", lambda: _config(tmp_path))
    monkeypatch.setattr(indexer, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(indexer, "iter_sources", lambda *_args, **_kwargs: [first, second])
    monkeypatch.setattr(
        indexer,
        "SentenceTransformer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("model must not load before source validation")
        ),
    )
    monkeypatch.setattr(
        indexer,
        "get_store",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("store must not open before source validation")
        ),
    )
    monkeypatch.setattr("sys.argv", ["rag-index"])

    with pytest.raises(ValueError, match="Duplicate source id 'json:duplicate'"):
        indexer.main()
