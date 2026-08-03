"""Pluggable retrieval store abstraction.

See ``docs/ADR-multi-corpus-profiles-and-pluggable-store.md`` (Axis 2).
``RetrievalStore`` is the Protocol every backend implements; ``ChromaStore``
is the default (and, for now, only) backend — behavior-preserving wrapper
around today's ChromaDB calls. ``get_store()`` is the factory every caller
(``query.py``, ``indexer.py``, ``indexing.py``, ``eval.py``, the API) uses to
open a store instead of importing a backend client library directly.
"""

from .base import RetrievalStore
from .chroma_store import ChromaStore
from . import chroma_store as _chroma

__all__ = [
    "RetrievalStore", "ChromaStore", "get_store",
    "list_collection_names", "drop_collection",
]


def _require_supported_backend(config: dict) -> None:
    """Guard: only the ``chroma`` backend is implemented today (the default when
    ``store`` is absent, matching ``config.yaml``). Any other value raises
    ``ValueError`` — e.g. ``"opensearch"`` is a future backend per the ADR,
    Axis 2, not built in this task."""
    backend = config.get("store", "chroma")
    if backend != "chroma":
        raise ValueError(
            f"Unknown store backend {backend!r} — only 'chroma' is implemented. "
            f"(See docs/ADR-multi-corpus-profiles-and-pluggable-store.md, Axis 2, "
            f"for the OpenSearchStore plan.)"
        )


def get_store(config: dict, collection_name: str = None):
    """Instantiate the ``RetrievalStore`` backend named in ``config['store']``.

    Only ``"chroma"`` is implemented today; any other value raises ``ValueError``.
    A store is bound to exactly one named collection — for backend-catalog
    operations that span collections, see :func:`list_collection_names` /
    :func:`drop_collection`.
    """
    _require_supported_backend(config)
    index_path = config.get("index_path", "./chroma_db")
    name = collection_name or config.get("collection_name", "obsidian_markdown")
    return ChromaStore(index_path, name)


def list_collection_names(config: dict) -> set:
    """Names of every collection in the configured index (backend-catalog op).

    Unlike :func:`get_store`'s single-collection view, this spans the whole
    backend catalog. Admin tooling only (``scripts/drop_collections.py``); the
    ``chromadb`` calls stay confined to ``chroma_store`` — this only dispatches
    on the configured backend.
    """
    _require_supported_backend(config)
    return _chroma.list_collection_names(config)


def drop_collection(config: dict, name: str) -> int:
    """Delete a collection by name, returning its pre-drop chunk count.

    Backend-catalog op (see :func:`list_collection_names`) — used by
    ``scripts/drop_collections.py`` to clear rejected trial collections.
    """
    _require_supported_backend(config)
    return _chroma.drop_collection(config, name)
