"""``ChromaStore`` — the ChromaDB-backed ``RetrievalStore`` implementation.

The ONLY module in this repo allowed to ``import chromadb`` (see the ADR,
Axis 2, and CLAUDE.md's call-site map). Behavior-preserving: reproduces
today's exact ChromaDB calls — ``PersistentClient`` + ``get_or_create_collection``
with ``metadata={"hnsw:space": "cosine"}``, content-hash ``upsert``,
``get(include=["metadatas"])`` for the incremental-diff snapshot, ``query()``
with the ``[0]``-unwrap, and ``delete()``. No ranking, metric, or ID change.
"""

import logging

# Import order matters: rag.utils sets ANONYMIZED_TELEMETRY and patches
# posthog.capture *before* chromadb is imported anywhere in the process (see
# utils.py's module docstring) — this is the only module that imports
# chromadb, so it is also the only place that ordering must be preserved.
from .. import utils  # noqa: F401

import chromadb

logger = logging.getLogger("rag")


class ChromaStore:
    """``RetrievalStore`` backed by one persistent ChromaDB collection."""

    def __init__(self, index_path: str, collection_name: str):
        self._client = chromadb.PersistentClient(
            path=index_path,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection_name = collection_name
        self._collection = None  # lazily opened; see _coll()

    @property
    def name(self) -> str:
        """The underlying collection's name."""
        if self._collection is not None:
            return self._collection.name
        return self._collection_name

    def _coll(self):
        """Return the open collection, opening it on first use if needed.

        Callers that never call :meth:`ensure` (query/eval/API — today's
        ``open_collection`` behavior) get ``get_collection``, which raises if
        the collection does not exist yet — matching current behavior
        exactly. ``ensure`` is the only path that creates one.
        """
        if self._collection is None:
            self._collection = self._client.get_collection(self._collection_name)
        return self._collection

    def ensure(self, name: str, dim=None, metric: str = "cosine") -> None:
        """Create the collection if missing; open it if it already exists.

        ``dim`` is accepted for interface parity but IGNORED for Chroma — it
        infers embedding dimensionality from the first upsert, never declared
        up front. ``metric`` is only honored on first creation: Chroma
        ignores ``hnsw:space`` metadata for a collection that already exists,
        so pointing ``ensure`` at a pre-existing collection never forces a
        rebuild (see the historical indexer.py comment this replaces).
        """
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": metric},
        )
        self._collection_name = name

    def snapshot(self) -> dict:
        """Return ``{id: metadata}`` for every chunk currently stored."""
        snap = self._coll().get(include=["metadatas"])
        return dict(zip(snap["ids"], snap["metadatas"]))

    def existing_ids(self) -> set:
        """Return the set of chunk IDs currently stored."""
        return set(self._coll().get(include=[])["ids"])

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        return self._coll().count()

    def upsert(self, ids, embeddings, docs, metas) -> None:
        """Insert-or-overwrite chunks by ID (embeddings + documents + metadata)."""
        self._coll().upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    def update_metadata(self, ids, metas) -> None:
        """Refresh metadata only for existing IDs — no re-embed."""
        self._coll().update(ids=ids, metadatas=metas)

    def delete(self, ids) -> None:
        """Remove chunks by ID."""
        self._coll().delete(ids=ids)

    def query(self, embedding, k, where=None) -> list:
        """Return the top-``k`` nearest records to ``embedding``, best-first.

        Mirrors the historical ``collection.query(...)`` call exactly: same
        ``include`` list, ``where`` only passed when truthy, and the ``[0]``-
        unwrap of Chroma's per-query-batch result shape (only one query
        embedding is ever passed here).
        """
        query_kwargs = dict(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._coll().query(**query_kwargs)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
