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

__all__ = ["RetrievalStore", "ChromaStore", "get_store"]


def get_store(config: dict, collection_name: str = None):
    """Instantiate the ``RetrievalStore`` backend named in ``config['store']``.

    Only ``"chroma"`` is implemented today (the default when ``store`` is
    absent from config, matching ``config.yaml``'s documented default). Any
    other value raises ``ValueError`` — e.g. ``"opensearch"`` is a future
    backend per the ADR, Axis 2, not built in this task.
    """
    backend = config.get("store", "chroma")
    if backend != "chroma":
        raise ValueError(
            f"Unknown store backend {backend!r} — only 'chroma' is implemented. "
            f"(See docs/ADR-multi-corpus-profiles-and-pluggable-store.md, Axis 2, "
            f"for the OpenSearchStore plan.)"
        )
    index_path = config.get("index_path", "./chroma_db")
    name = collection_name or config.get("collection_name", "obsidian_markdown")
    return ChromaStore(index_path, name)
