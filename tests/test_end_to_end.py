"""End-to-end validation of the hardened pipeline — fully offline.

Chains the subsystems the hardening plan wired together, using a fake embedding
model and a real ChromaStore over a temp dir (no model download, no network, no
real corpus):

  config preflight → atomic artifact publication → safe indexing (provenance) →
  lexical sidecar freshness → retrieval through search() → backup verify + restore
  drill → API readiness + an authorized /query.

The recall@k/MRR eval-vs-baseline gate is intentionally NOT run here: it needs a
populated real index and the real model, so it stays a manual `make eval` step
(documented in docs/configuration.md and the eval harness).
"""

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from extractor import artifacts
from extractor.build_sqlite import build_database
from rag import backup, query as rag_query
from rag.api.app import app
from rag.api.auth import JWT_SECRET_ENV
from rag.api.deps import get_rag_state
from rag.api.token import mint_token
from rag.lexical import LexicalIndex, LexicalIndexStaleError, get_lexical
from rag.provenance import (
    IndexProvenance,
    IndexState,
    begin_generation,
    read_index_state,
)
from rag.store.chroma_store import ChromaStore
from rag.utils import apply_offline_mode

COLLECTION = "e2e"
DIM = 4


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer (dim-4, offline)."""

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, **kw):
        # A cheap, stable embedding: bag-of-length features. Never random.
        rows = []
        for text in texts:
            n = float(len(text or ""))
            rows.append([n % 3, (n + 1) % 3, (n + 2) % 3, 1.0])
        return np.array(rows, dtype="float32")


def _provenance():
    return IndexProvenance(
        embedding_model="fake-model",
        embedding_revision="",
        embedding_dimension=DIM,
        normalized=True,
        chunker_version="heading-paragraph-v1",
        chunk_max_chars=1200,
        chunk_overlap_chars=150,
        metric="cosine",
        corpus_profile="e2e",
    )


def _publish_artifact_generation(output_root):
    """Publish one complete, validated artifact generation and return its path."""
    gen = artifacts.begin_generation(output_root)
    for rel in ("text/books/manifest.json", "text/resources/manifest.json",
                "indexed/build_report.json"):
        p = gen.staging / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]" if p.name == "manifest.json" else "{}", encoding="utf-8")
    catalog = gen.staging / "catalog/resource_inventory_enriched.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("", encoding="utf-8")
    idx = gen.staging / "indexed/index_documents.jsonl"
    vault = gen.staging / "indexed/vault_documents.jsonl"
    idx.write_text("", encoding="utf-8")
    vault.write_text("", encoding="utf-8")
    build_database([idx, vault], gen.staging / "resources.db")
    return artifacts.publish_generation(gen, profile="e2e")


def test_end_to_end_offline_pipeline(tmp_path, monkeypatch):
    # ── 1. Config preflight ──────────────────────────────────────────────────
    index_path = tmp_path / "chroma"
    lexical_path = tmp_path / "lexical" / f"{COLLECTION}.db"
    config_data = {
        "vault_path": str(tmp_path / "vault"),
        "index_path": str(index_path),
        "lexical_path": str(lexical_path),
        "collection_name": COLLECTION,
        "store": "chroma",
        "embedding_model": "fake-model",
        "embedding_dimension": DIM,
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 150,
        "embedding_batch_size": 16,
        "markdown_workers": 1,
        "pdf_workers": 1,
        "offline": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    from rag.config import load_config
    config = load_config(config_path)
    assert config["collection_name"] == COLLECTION
    # offline enforcement flips the HF flags before any model load
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    assert apply_offline_mode(config) is True

    # ── 2. Atomic artifact publication ───────────────────────────────────────
    output_root = tmp_path / "artifacts"
    final = _publish_artifact_generation(output_root)
    assert artifacts.resolve_active(output_root) == final
    assert (output_root / "current").is_symlink()

    # ── 3. Safe indexing (store + provenance generation) ─────────────────────
    store = ChromaStore(str(index_path), COLLECTION)
    store.ensure(COLLECTION)
    model = _FakeModel()
    docs = [
        "kubernetes pod networking and ingress",
        "python packaging with pyproject and wheels",
        "sqlite fts5 full text search",
    ]
    ids = [f"c{i}" for i in range(len(docs))]
    embeddings = model.encode(docs).tolist()
    metas = [{"title": f"doc{i}", "path": f"p{i}.md", "domain": "DevOps"} for i in range(len(docs))]
    store.upsert(ids, embeddings, docs, metas)
    begin_generation(store, _provenance())
    state = read_index_state(store)
    assert state is not None and store.count() == 3

    # ── 4. Lexical sidecar freshness ─────────────────────────────────────────
    n = LexicalIndex.build(
        store.iter_records(), lexical_path, index_state=state,
        expected_count=store.count(), state_reader=lambda: read_index_state(store),
    )
    assert n == 3
    # A matching state resolves; a different generation is rejected as stale.
    assert get_lexical(config, COLLECTION, index_state=state).query("kubernetes", 3)
    stale = IndexState.create(state.provenance)
    with pytest.raises(LexicalIndexStaleError):
        get_lexical(config, COLLECTION, index_state=stale)

    # ── 5. Retrieval through the single seam ─────────────────────────────────
    records = rag_query.search(
        "kubernetes networking", n_results=3, config=config, model=model, store=store
    )
    assert records and all("document" in r and "rank" in r for r in records)

    # ── 6. Backup verify + restore drill ─────────────────────────────────────
    backup_dir = backup.create_backup(config, tmp_path / "backup", COLLECTION)
    assert backup.verify_backup(backup_dir)["chunk_count"] == 3
    drill = backup.restore_drill(backup_dir)
    assert drill["chunk_count"] == 3 and drill["provenance_verified"] is True

    # ── 7. API readiness + an authorized query ───────────────────────────────
    monkeypatch.setenv(JWT_SECRET_ENV, "e2e-secret-key-at-least-32-bytes-long!!")
    state_dict = {
        "config": config,
        "model": model,
        "store": store,
        "reranker": None,
        "generator": None,
        "embedding_model": "fake-model",
        "reranker_model": "fake-rerank",
        "inference_limiter": None,
    }
    app.dependency_overrides[get_rag_state] = lambda: state_dict
    try:
        client = TestClient(app)
        ready = client.get("/ready")
        assert ready.status_code == 200 and ready.json()["ready"] is True
        assert ready.json()["generation_id"] == state.generation_id

        token = mint_token("e2e", 1, scopes=["query"])
        resp = client.post(
            "/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "kubernetes networking", "n_results": 3},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1
        # A query-scoped token cannot trigger indexing.
        assert client.post(
            "/index", headers={"Authorization": f"Bearer {token}"}
        ).status_code == 403
    finally:
        app.dependency_overrides.clear()
