"""``RetrievalStore`` — the pluggable-backend protocol.

See ``docs/ADR-multi-corpus-profiles-and-pluggable-store.md`` (Axis 2). Every
caller in this codebase (``query.py``, ``indexer.py``, ``indexing.py``,
``eval.py``, the API) depends on this Protocol instead of importing a backend
client library directly, so a store swap (e.g. OpenSearch, later) never
touches the retrieval/indexing engine.

The method set here is richer than the ADR's 6-method sketch — it also covers
the metadata-only refresh path (``update_metadata``, distinct from
``upsert`` so a metadata-only edit is never mistaken for a re-embed) and the
full-metadata ``snapshot`` the incremental indexer needs for its new/update/
skip diff (``existing_ids`` alone is not enough for that).
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class RetrievalStore(Protocol):
    """Uniform interface over one named vector/lexical collection.

    A store instance owns exactly one collection/index (named at construction
    time via ``get_store()``). ``ChromaStore`` is the default, behavior-
    preserving implementation; see its docstring for the exact ChromaDB calls
    each method wraps.
    """

    @property
    def name(self) -> str:
        """The underlying collection's name."""
        ...

    def ensure(self, name: str, dim: Optional[int] = None, metric: str = "cosine") -> None:
        """Create the collection if it does not exist yet; open it if it does.

        ``dim`` is accepted for interface parity across backends but may be
        ignored by a backend (e.g. Chroma infers dimensionality from the
        first upsert). ``metric`` only takes effect on first creation for
        backends where the metric is fixed at collection-creation time.
        """
        ...

    def snapshot(self) -> dict:
        """Return the full ``{chunk_id: metadata}`` map currently stored.

        Used by the incremental indexer to classify every candidate chunk as
        new / metadata-changed / unchanged without a second round-trip.
        """
        ...

    def existing_ids(self) -> set:
        """Return the set of chunk IDs currently stored.

        Must reflect the same population as :meth:`count`. Provided for
        ID-only callers; the incremental indexer's 0-files anti-wipe guard
        derives its ID set from :meth:`snapshot` instead.
        """
        ...

    def count(self) -> int:
        """Return the number of chunks currently stored."""
        ...

    def upsert(self, ids: list, embeddings: list, docs: list, metas: list) -> None:
        """Insert-or-overwrite chunks by ID (embeddings + documents + metadata)."""
        ...

    def update_metadata(self, ids: list, metas: list) -> None:
        """Refresh metadata only for existing IDs — no re-embed.

        Distinct from :meth:`upsert` on purpose: folding this into ``upsert``
        would force embeddings to be recomputed for a metadata-only change
        (e.g. an edited frontmatter field with unchanged chunk body).
        """
        ...

    def delete(self, ids: list) -> None:
        """Remove chunks by ID."""
        ...

    def query(self, embedding: list, k: int, where: Optional[dict] = None) -> list:
        """Return the top-``k`` nearest records to ``embedding``, best-first.

        Each record is a dict with keys ``document`` (str), ``metadata``
        (dict) and ``distance`` (float). ``where`` is a backend-native
        metadata filter (see ``query.build_where`` for Chroma's dict shape);
        pass ``None``/empty for no filter. Rank assignment and any rerank
        step happen ABOVE this call, in ``query.search()`` — a store never
        reranks.
        """
        ...
