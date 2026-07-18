"""Shared FastAPI dependencies — accessors for the once-loaded app state.

The heavy objects (embedding model, Chroma collection, reranker) are loaded a
single time in ``app.lifespan`` and stashed on ``app.state.rag``. Routes read
them through :func:`get_rag_state` rather than reloading per request.
"""

from fastapi import Request


def get_rag_state(request: Request) -> dict:
    """Return the shared RAG state dict stashed on ``app.state`` at startup.

    Keys: ``config``, ``model``, ``collection``, ``reranker``,
    ``embedding_model``, ``reranker_model``.
    """
    return request.app.state.rag
