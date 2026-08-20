"""Unit tests for the incremental engine (rag.indexing) — new/update/skip,
stale preservation, and end-to-end idempotency through the source registry.

Uses a fake embedding model and a real ChromaStore backed by a temporary
on-disk ChromaDB (via the pluggable RetrievalStore seam — see
docs/ADR-multi-corpus-profiles-and-pluggable-store.md, Axis 2), so it runs
offline with no GPU and no real index."""

import tempfile
from pathlib import Path

import numpy as np

from rag.extractors import iter_sources
from rag.indexing import (
    _bounded_extractions,
    index_file_chunks,
    preserve_existing,
    run_source,
)
from rag.reconciliation import ReconciliationCatalog
from rag.store.chroma_store import ChromaStore


class FakeModel:
    """Stand-in for SentenceTransformer — shape-correct, content-irrelevant."""
    def encode(self, docs, normalize_embeddings=True, batch_size=16):
        return np.zeros((len(docs), 4))


def _store(name="unit"):
    store = ChromaStore(tempfile.mkdtemp(), name)
    store.ensure(name)
    return store


def _catalog(store):
    return ReconciliationCatalog.from_store(
        store,
        Path(tempfile.mkdtemp()) / "index",
        page_size=2,
    )


def _metadata(store):
    return dict(store.iter_metadata(page_size=2))


def test_index_file_chunks_new_update_skip():
    store = _store("nus")
    model = FakeModel()
    ids = ["i1", "i2"]
    docs = ["alpha", "beta"]
    metas = [{"path": "p", "domain": "DevOps"}, {"path": "p", "domain": "DevOps"}]

    with _catalog(store) as reconciliation:
        # first pass: both new -> embedded
        assert index_file_chunks(
            ids, docs, metas, reconciliation, model, "cpu", 16, store
        ) == (2, 0, 2)

        # identical content + metadata -> skipped
        assert index_file_chunks(
            ids, docs, metas, reconciliation, model, "cpu", 16, store
        ) == (0, 0, 2)

        # same body, changed metadata -> metadata refresh only (no re-embed)
        changed = [
            {"path": "p", "domain": "Platform"},
            {"path": "p", "domain": "Platform"},
        ]
        assert index_file_chunks(
            ids, docs, changed, reconciliation, model, "cpu", 16, store
        ) == (0, 2, 2)


def test_preserve_existing_marks_only_matching_owned_file():
    store = _store("preserve")
    store.upsert(
        ["a", "b", "c"],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        ["a", "b", "c"],
        [
            {"source_id": "json:catalog", "file_id": "x"},
            {"source_id": "json:catalog", "file_id": "y"},
            {"source_id": "json:catalog", "file_id": "x"},
        ],
    )
    with _catalog(store) as reconciliation:
        assert preserve_existing("json:catalog", "x", reconciliation) == 2
        assert reconciliation.source_counts("json:catalog") == (3, 2, 1)


def test_run_source_end_to_end_is_idempotent(tmp_path):
    (tmp_path / "Knowledge").mkdir()
    (tmp_path / "Knowledge" / "n.md").write_text(
        "---\ntitle: N\ndomain: DevOps\n---\n# H\n" + "word " * 300, encoding="utf-8")
    (tmp_path / "Templates").mkdir()
    (tmp_path / "Templates" / "t.md").write_text("---\ntitle: T\n---\nignored " * 50, encoding="utf-8")

    config = {
        "vault_path": str(tmp_path), "exclude_dirs": ["Templates"], "exclude_files": [],
        "markdown_workers": 1, "pdf_workers": 1,
    }
    store = _store("e2e")
    model = FakeModel()

    def run():
        new = updated = stale = 0
        with _catalog(store) as reconciliation:
            for source in iter_sources(config, tmp_path, 1200, 150):
                _, source_new, source_updated = run_source(
                    source, reconciliation, model, "cpu", 16, store
                )
                new += source_new
                updated += source_updated
                stale += reconciliation.source_counts(source.source_id)[2]
        return new, updated, stale

    first = run()
    second = run()
    assert first[0] > 0          # first run embeds the (non-excluded) note
    assert second == (0, 0, 0)   # nothing changed -> no work, nothing pruned


def test_pdf_sources_fall_back_behind_indexable_json_coverage(tmp_path):
    """Only valid JSON from the matching source group suppresses live parsing."""
    import json

    pdf_dir = tmp_path / "Books"
    pdf_dir.mkdir()
    (pdf_dir / "covered.pdf").write_bytes(b"%PDF-1.4")
    (pdf_dir / "new-arrival.pdf").write_bytes(b"%PDF-1.4")
    json_dir = tmp_path / "indexed"
    json_dir.mkdir()
    (json_dir / "covered.json").write_text(
        json.dumps(
            {
                "file_name": "covered.pdf",
                "source_group": "books",
                "title": "Covered",
                "text": "indexable text " * 10,
            }
        ),
        encoding="utf-8",
    )

    config = {
        "vault_path": str(tmp_path),
        "exclude_dirs": [],
        "exclude_files": [],
        "pdf_sources": [{"path": str(pdf_dir), "type": "book"}],
        "json_sources": [{"path": str(json_dir)}],
    }
    sources = {s.kind: s for s in iter_sources(config, tmp_path, 1200, 150)}
    assert [path.name for path in sources["pdf"].files] == ["new-arrival.pdf"]
    assert sources["pdf"].excluded_files == 1
    assert "covered by JSON, skipped" in sources["pdf"].label
    assert len(sources["json"].files) == 1


def test_short_json_does_not_suppress_pdf_fallback(tmp_path):
    import json

    pdf_dir = tmp_path / "Books"
    pdf_dir.mkdir()
    (pdf_dir / "book.pdf").write_bytes(b"%PDF-1.4")
    json_dir = tmp_path / "indexed"
    json_dir.mkdir()
    (json_dir / "book.json").write_text(
        json.dumps({"file_name": "book.pdf", "source_group": "books", "text": "short"}),
        encoding="utf-8",
    )
    config = {
        "exclude_dirs": [],
        "exclude_files": [],
        "pdf_sources": [{"path": str(pdf_dir), "type": "book"}],
        "json_sources": [{"path": str(json_dir)}],
    }

    pdf_source = next(
        source for source in iter_sources(config, tmp_path, 1200, 150) if source.kind == "pdf"
    )

    assert [path.name for path in pdf_source.files] == ["book.pdf"]
    assert pdf_source.excluded_files == 0


def test_json_coverage_is_scoped_to_matching_pdf_group(tmp_path):
    import json

    books = tmp_path / "Books"
    resources = tmp_path / "Resources"
    indexed = tmp_path / "indexed"
    books.mkdir()
    resources.mkdir()
    indexed.mkdir()
    (books / "shared.pdf").write_bytes(b"%PDF-1.4")
    (resources / "shared.pdf").write_bytes(b"%PDF-1.4")
    (indexed / "shared.json").write_text(
        json.dumps(
            {
                "file_name": "shared.pdf",
                "source_group": "books",
                "text": "indexable text " * 10,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "exclude_dirs": [],
        "exclude_files": [],
        "pdf_sources": [
            {"id": "pdf:book", "path": str(books), "type": "book"},
            {"id": "pdf:resource", "path": str(resources), "type": "resource"},
        ],
        "json_sources": [{"path": str(indexed)}],
    }

    pdf_sources = {
        source.source_id: source
        for source in iter_sources(config, tmp_path, 1200, 150)
        if source.kind == "pdf"
    }

    assert pdf_sources["pdf:book"].files == []
    assert pdf_sources["pdf:book"].excluded_files == 1
    assert [path.name for path in pdf_sources["pdf:resource"].files] == ["shared.pdf"]


def test_all_pdfs_delegated_to_json_produces_reviewable_prune_plan(tmp_path):
    import json

    from rag.index_manifest import build_prune_plan

    books = tmp_path / "Books"
    indexed = tmp_path / "indexed"
    books.mkdir()
    indexed.mkdir()
    (books / "covered.pdf").write_bytes(b"%PDF-1.4")
    (indexed / "covered.json").write_text(
        json.dumps(
            {
                "file_name": "covered.pdf",
                "source_group": "books",
                "text": "indexable text " * 10,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "exclude_dirs": [],
        "exclude_files": [],
        "pdf_sources": [{"id": "pdf:book", "path": str(books), "type": "book"}],
        "json_sources": [{"path": str(indexed)}],
    }
    source = next(
        item for item in iter_sources(config, tmp_path, 1200, 150) if item.kind == "pdf"
    )
    store = _store("delegated")
    store.upsert(
        ["old-pdf-chunk"],
        [[1, 0, 0, 0]],
        ["old"],
        [{"source_id": "pdf:book", "file_id": "covered.pdf"}],
    )
    with _catalog(store) as reconciliation:
        result = run_source(
            source, reconciliation, FakeModel(), "cpu", 16, store
        )
        blocked = build_prune_plan(
            reconciliation, [result], max_fraction=0.25, max_chunks=100
        )
        approved = build_prune_plan(
            reconciliation,
            [result],
            max_fraction=0.25,
            max_chunks=100,
            allow_large_prune=True,
        )

        assert result.state == "completed"
        assert result.file_count == 1
        assert result.prune_eligible
        assert blocked.approved_deletion_count == 0
        assert "threshold exceeded" in blocked.sources[0].reason
        assert approved.approved_deletion_count == 1


def test_run_source_adds_source_and_file_ownership(tmp_path):
    from rag.extractors import Source

    source_file = tmp_path / "doc.json"
    source_file.write_text("{}")
    source = Source(
        source_id="json:catalog",
        kind="json",
        label="JSON",
        root=tmp_path,
        files=[source_file],
        workers=1,
        extract=lambda _path: (
            ["chunk"],
            ["body"],
            [{"path": "book.pdf"}],
            None,
        ),
        file_key=lambda path: path.name,
    )
    store = _store("ownership")
    with _catalog(store) as reconciliation:
        result = run_source(
            source, reconciliation, FakeModel(), "cpu", 16, store
        )

    metadata = _metadata(store)["chunk"]
    assert metadata["source_id"] == "json:catalog"
    assert metadata["source_kind"] == "json"
    assert metadata["file_id"] == "doc.json"
    assert result.state == "completed"
    assert result.prune_eligible


def test_run_source_degrades_and_preserves_owned_chunk_on_error(tmp_path):
    from rag.extractors import Source

    source_file = tmp_path / "bad.json"
    source_file.write_text("bad")
    source = Source(
        source_id="json:catalog",
        kind="json",
        label="JSON",
        root=tmp_path,
        files=[source_file],
        workers=1,
        extract=lambda _path: ([], [], [], "json read error"),
        file_key=lambda path: path.name,
    )
    store = _store("failed")
    store.upsert(
        ["old"],
        [[1, 0, 0, 0]],
        ["old body"],
        [
            {
                "source_id": "json:catalog",
                "file_id": "bad.json",
                "path": "book.pdf",
            }
        ],
    )
    with _catalog(store) as reconciliation:
        result = run_source(
            source, reconciliation, FakeModel(), "cpu", 16, store
        )
        counts = reconciliation.source_counts("json:catalog")

    assert result.state == "degraded"
    assert result.failed_files == 1
    assert not result.prune_eligible
    assert counts == (1, 1, 0)


def test_run_source_dry_run_classifies_without_store_mutations(tmp_path):
    from rag.extractors import Source

    source_file = tmp_path / "note.md"
    source_file.write_text("body")
    source = Source(
        source_id="markdown:vault",
        kind="markdown",
        label="Markdown",
        root=tmp_path,
        files=[source_file],
        workers=1,
        extract=lambda _path: (
            ["new", "existing"],
            ["new body", "old body"],
            [{"path": "note.md"}, {"path": "note.md", "domain": "new"}],
            None,
        ),
        file_key=lambda path: path.name,
    )

    class ReadOnlyStore:
        def upsert(self, **_kwargs):
            raise AssertionError("dry run must not upsert")

        def update_metadata(self, **_kwargs):
            raise AssertionError("dry run must not update metadata")

    seed_store = _store("dry-run-seed")
    seed_store.upsert(
        ["existing"],
        [[1, 0, 0, 0]],
        ["old body"],
        [
            {
                "path": "note.md",
                "domain": "old",
                "source_id": "markdown:vault",
                "source_kind": "markdown",
                "file_id": "note.md",
            }
        ],
    )
    with _catalog(seed_store) as reconciliation:
        result = run_source(
            source,
            reconciliation,
            None,
            "cpu",
            16,
            ReadOnlyStore(),
            dry_run=True,
        )
        counts = reconciliation.source_counts("markdown:vault")

    assert result.new_chunks == 1
    assert result.updated_chunks == 1
    assert counts == (1, 2, 0)


def test_bounded_extractions_never_exceeds_worker_window(tmp_path, monkeypatch):
    from rag.extractors import Source
    import rag.indexing as indexing

    class Future:
        def __init__(self, executor, value):
            self.executor = executor
            self.value = value

        def result(self):
            self.executor.outstanding -= 1
            return self.value

    class Executor:
        maximum = 0

        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.outstanding = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def submit(self, function, path):
            self.outstanding += 1
            type(self).maximum = max(type(self).maximum, self.outstanding)
            assert self.outstanding <= self.max_workers
            return Future(self, function(path))

    files = []
    for index in range(20):
        path = tmp_path / f"{index}.md"
        path.write_text("body")
        files.append(path)
    source = Source(
        source_id="markdown:vault",
        kind="markdown",
        label="Markdown",
        root=tmp_path,
        files=files,
        workers=3,
        extract=lambda path: ([], [path.name], [], None),
        file_key=lambda path: path.name,
    )
    monkeypatch.setattr(indexing, "ThreadPoolExecutor", Executor)

    results = list(_bounded_extractions(source))

    assert len(results) == 20
    assert Executor.maximum == 3
