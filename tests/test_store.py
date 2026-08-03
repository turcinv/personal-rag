"""Tests for the multi-corpus config profiles + pluggable store refactor
(docs/ADR-multi-corpus-profiles-and-pluggable-store.md).

Axis 1 (config profiles) is covered here now — profile-loading tests only,
offline, no chromadb/model involved. Axis 2 (RetrievalStore/ChromaStore
parity tests against a temp PersistentClient) lands later in this same file;
keep new sections clearly separated so both can coexist.
"""

import tempfile
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

# Path overrides that load_config() applies on top of whatever profile is
# selected. rag.utils calls load_dotenv() at import, so a developer's real .env
# leaks into these assertions (RAG_JSON_PATH in particular REPLACES the whole
# json_sources list, which silently breaks the markdown-only logmanager profile).
# Clearing the shell env is not enough — dotenv repopulates from the file — so
# each profile test drops them from os.environ after import.
_PATH_OVERRIDES = (
    "RAG_VAULT_PATH", "RAG_PDF_BOOKS_PATH", "RAG_PDF_RESOURCES_PATH",
    "RAG_JSON_PATH", "RAG_INDEX_PATH",
)


def _isolate_path_overrides(monkeypatch):
    for var in _PATH_OVERRIDES:
        monkeypatch.delenv(var, raising=False)


def test_personal_profile_loads_expected_values(monkeypatch):
    _isolate_path_overrides(monkeypatch)
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
    _isolate_path_overrides(monkeypatch)
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
    _isolate_path_overrides(monkeypatch)
    monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / "config.yaml"))

    cfg = load_config()

    assert cfg["store"] == "chroma"


def test_profiles_carry_expected_rerank_default(monkeypatch):
    """rerank_default is per-profile: off for the personal corpus (measured
    2026-07-27 — rerank loses overall recall@5 there), left on for the wiki
    profile until there is a wiki eval set to measure against."""
    for profile, expected in [
        ("config.yaml", False),
        ("config.personal.yaml", False),
        ("config.logmanager.yaml", True),
    ]:
        _isolate_path_overrides(monkeypatch)
        monkeypatch.setenv("RAG_CONFIG_PATH", str(REPO_ROOT / profile))
        assert load_config()["rerank_default"] is expected, profile


# ─────────────────────────────────────────────────────────────────────────────
# Axis 2 — store
# ─────────────────────────────────────────────────────────────────────────────
#
# ChromaStore parity against a temp PersistentClient (reusing the pattern from
# tests/test_indexing.py's _collection()), the 0-files anti-wipe guard driven
# via the store signal, and the no-chromadb-leak import check.

import chromadb

from rag.store.chroma_store import ChromaStore
from rag.store import get_store, list_collection_names, drop_collection


def _raw_collection(name):
    """Pre-refactor direct-chromadb path: a bare PersistentClient collection
    created exactly like the historical indexer.py/query.py call —
    ``get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`` —
    which is also exactly what ChromaStore.ensure() does under the hood."""
    client = chromadb.PersistentClient(
        path=tempfile.mkdtemp(),
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def _store(name):
    store = ChromaStore(tempfile.mkdtemp(), name)
    store.ensure(name)
    return store


def test_chroma_store_name_property():
    store = ChromaStore(tempfile.mkdtemp(), "before-ensure")
    assert store.name == "before-ensure"   # pre-creation: falls back to the ctor arg
    store.ensure("after-ensure")
    assert store.name == "after-ensure"    # post-creation: reflects the open collection


def test_chroma_store_ensure_upsert_count_existing_ids_snapshot():
    store = _store("parity-basics")
    ids = ["a", "b"]
    embeddings = [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
    docs = ["doc a", "doc b"]
    metas = [{"path": "p1"}, {"path": "p2"}]

    store.upsert(ids, embeddings, docs, metas)

    assert store.count() == 2
    assert store.existing_ids() == {"a", "b"}
    assert store.snapshot() == {"a": {"path": "p1"}, "b": {"path": "p2"}}


def test_chroma_store_upsert_query_matches_direct_chromadb_path():
    """upsert then query returns the same records/order as the pre-refactor
    direct-chromadb path (same PersistentClient calls under the hood).

    Vectors are chosen with no component in common so cosine distances to the
    query come out strictly ordered (no ties) — the test asserts a specific
    order deterministically, not just "some order"."""
    ids = ["a", "b", "c", "d"]
    embeddings = [
        [1.0, 0.2, 0.0, 0.0],
        [0.8, 0.6, 0.1, 0.0],
        [0.3, 0.9, 0.2, 0.1],
        [0.0, 0.1, 1.0, 0.2],
    ]
    docs = ["doc a", "doc b", "doc c", "doc d"]
    metas = [{"path": f"p{i}", "domain": "X" if i % 2 == 0 else "Y"} for i in range(4)]
    query_vec = [1.0, 0.0, 0.0, 0.0]

    # Pre-refactor direct-chromadb path.
    raw = _raw_collection("direct")
    raw.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    raw_results = raw.query(
        query_embeddings=[query_vec], n_results=3,
        include=["documents", "metadatas", "distances"],
    )
    raw_records = [
        {"document": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(
            raw_results["documents"][0], raw_results["metadatas"][0], raw_results["distances"][0],
        )
    ]

    # ChromaStore path.
    store = _store("via-store")
    store.upsert(ids, embeddings, docs, metas)
    store_records = store.query(query_vec, 3)

    assert store_records == raw_records
    assert [r["document"] for r in store_records] == ["doc a", "doc b", "doc c"]

    # Phase 1: text=/hybrid= are accepted for interface parity but IGNORED by
    # Chroma — passing them must return byte-identical records (and never add an
    # `id` key), so dense-only behaviour and the eval baseline stay unchanged.
    assert store.query(query_vec, 3, text="doc a", hybrid=True) == store_records


def test_chroma_store_query_honors_where_filter():
    store = _store("parity-where")
    ids = ["a", "b", "c"]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    docs = ["doc a", "doc b", "doc c"]
    metas = [{"domain": "X"}, {"domain": "Y"}, {"domain": "X"}]
    store.upsert(ids, embeddings, docs, metas)

    hits = store.query([1.0, 0.0, 0.0, 0.0], 5, where={"domain": {"$eq": "X"}})
    assert {h["document"] for h in hits} == {"doc a", "doc c"}
    assert all(h["metadata"]["domain"] == "X" for h in hits)


def test_chroma_store_iter_records_yields_id_document_metadata():
    """iter_records() is the full-fidelity read (id + document + metadata) the
    lexical-index builder consumes — snapshot() carries metadata only."""
    store = _store("parity-iter")
    ids = ["a", "b", "c"]
    embeddings = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]]
    docs = ["doc a", "doc b", "doc c"]
    metas = [{"path": "p1"}, {"path": "p2"}, {"path": "p3"}]
    store.upsert(ids, embeddings, docs, metas)

    got = {cid: (doc, meta) for cid, doc, meta in store.iter_records()}
    assert got == {
        "a": ("doc a", {"path": "p1"}),
        "b": ("doc b", {"path": "p2"}),
        "c": ("doc c", {"path": "p3"}),
    }


def test_chroma_store_iter_records_pages_over_all_chunks():
    """Paging must return every chunk even when the page size is smaller than the
    collection (the Jetson-memory-safe path), not just the first page."""
    store = _store("parity-iter-paged")
    n = 25
    store.upsert(
        ids=[f"id{i}" for i in range(n)],
        embeddings=[[float(i), 1.0, 0, 0] for i in range(n)],
        docs=[f"doc {i}" for i in range(n)],
        metas=[{"path": f"p{i}"} for i in range(n)],
    )
    seen = list(store.iter_records(page_size=10))   # 3 pages: 10 + 10 + 5
    assert len(seen) == n
    assert {cid for cid, _, _ in seen} == {f"id{i}" for i in range(n)}


def test_chroma_store_existing_ids_returns_upserted_ids():
    store = _store("parity-ids")
    store.upsert(
        ids=["x1", "x2", "x3"],
        embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        docs=["d1", "d2", "d3"],
        metas=[{"path": "p1"}, {"path": "p2"}, {"path": "p3"}],
    )
    assert store.existing_ids() == {"x1", "x2", "x3"}


def test_chroma_store_update_metadata_refreshes_without_reembed_or_count_change():
    store = _store("parity-updatemeta")
    ids = ["a", "b"]
    embeddings = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    docs = ["doc a", "doc b"]
    metas = [{"path": "p1", "domain": "DevOps"}, {"path": "p2", "domain": "DevOps"}]
    store.upsert(ids, embeddings, docs, metas)

    before = store._coll().get(ids=ids, include=["embeddings"])
    before_embeddings = {i: list(e) for i, e in zip(before["ids"], before["embeddings"])}

    new_metas = [{"path": "p1", "domain": "Platform"}, {"path": "p2", "domain": "Platform"}]
    store.update_metadata(ids, new_metas)

    # Metadata changed...
    assert store.snapshot() == {"a": new_metas[0], "b": new_metas[1]}
    # ...but count is stable and the embeddings were never touched (no re-embed).
    assert store.count() == 2
    after = store._coll().get(ids=ids, include=["embeddings"])
    after_embeddings = {i: list(e) for i, e in zip(after["ids"], after["embeddings"])}
    assert after_embeddings == before_embeddings


def test_chroma_store_delete_prunes_count_and_existing_ids():
    store = _store("parity-delete")
    store.upsert(
        ids=["a", "b", "c"],
        embeddings=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        docs=["da", "db", "dc"],
        metas=[{"path": "p"}, {"path": "p"}, {"path": "p"}],
    )
    assert store.count() == 3

    store.delete(["b"])

    assert store.count() == 2
    assert store.existing_ids() == {"a", "c"}


# ── 0-files anti-wipe guard, driven via the store signal ─────────────────────────


import rag.indexer as indexer_mod  # noqa: E402  (kept with this section)


class _FakeSentenceTransformer:
    """Never loads a real model — the guard fires before the model is used."""
    def __init__(self, *a, **kw):
        pass


def test_indexer_main_raises_and_does_not_prune_when_zero_files_but_store_nonempty(
    tmp_path, monkeypatch,
):
    """Incident 2026-07-15 guard, re-asserted through the store seam: if every
    source reports 0 files while store.existing_ids()/count() is non-empty,
    indexer.main() must raise RuntimeError and must NOT delete anything."""
    index_path = tmp_path / "chroma"
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    collection_name = "guard-test"

    # Seed the store exactly like a real prior index run would have left it.
    seed_store = ChromaStore(str(index_path), collection_name)
    seed_store.ensure(collection_name)
    seed_store.upsert(
        ids=["x1", "x2"],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        docs=["doc x1", "doc x2"],
        metas=[{"path": "p1"}, {"path": "p2"}],
    )
    assert seed_store.count() == 2

    fake_config = {
        "vault_path": str(vault_path),
        "collection_name": collection_name,
        "index_path": str(index_path),
        "store": "chroma",
    }

    monkeypatch.setattr(indexer_mod, "load_config", lambda: fake_config)
    monkeypatch.setattr(indexer_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(indexer_mod, "SentenceTransformer", _FakeSentenceTransformer)
    monkeypatch.setattr(indexer_mod, "iter_sources", lambda *a, **k: [])  # 0 files, every source
    monkeypatch.setattr("sys.argv", ["rag-index"])

    with pytest.raises(RuntimeError, match="0 files"):
        indexer_mod.main()

    # No pruning happened: re-open the store fresh and confirm the seeded
    # chunks are still exactly there.
    verify_store = ChromaStore(str(index_path), collection_name)
    verify_store.ensure(collection_name)
    assert verify_store.count() == 2
    assert verify_store.existing_ids() == {"x1", "x2"}


# ── backend-catalog helpers (list / drop collections) ───────────────────────────
#
# Used by scripts/drop_collections.py — operations a single-collection ChromaStore
# can't express (they span the index's whole collection catalog).


def test_catalog_helpers_list_and_drop_collections():
    """list_collection_names sees every collection in one index dir; drop_collection
    returns the pre-drop chunk count and removes only the named collection."""
    index_path = tempfile.mkdtemp()
    config = {"store": "chroma", "index_path": index_path, "collection_name": "col_a"}

    # Two collections in the SAME index dir (a ChromaStore only ever binds one).
    for name in ("col_a", "col_b"):
        store = get_store(config, collection_name=name)
        store.ensure(name)
        store.upsert(["x"], [[1.0, 0.0, 0.0, 0.0]], ["doc"], [{"path": "p"}])

    assert list_collection_names(config) == {"col_a", "col_b"}

    dropped = drop_collection(config, "col_b")
    assert dropped == 1                                   # pre-drop chunk count
    assert list_collection_names(config) == {"col_a"}     # only col_b removed


def test_catalog_helpers_reject_unsupported_backend():
    """Same backend gate as get_store — only 'chroma' is implemented."""
    config = {"store": "opensearch", "index_path": tempfile.mkdtemp()}
    with pytest.raises(ValueError):
        list_collection_names(config)
    with pytest.raises(ValueError):
        drop_collection(config, "whatever")


# ── no chromadb leak outside the store package ───────────────────────────────────


def test_chromadb_import_confined_to_chroma_store_module():
    """Only src/rag/store/chroma_store.py may import/use chromadb — every other
    module in the rag engine package must go through the RetrievalStore seam.
    (scripts/ and tests/ live outside src/rag/ and are not scoped by this check;
    both former direct-chromadb callers — scripts/drop_collections.py and
    tests/test_queries.py — now route through the rag.store package too, via
    get_store / list_collection_names / drop_collection.)"""
    rag_src = REPO_ROOT / "src" / "rag"
    allowed = rag_src / "store" / "chroma_store.py"

    offenders = []
    for path in sorted(rag_src.rglob("*.py")):
        if path == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "import chromadb" in text or "chromadb." in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []
