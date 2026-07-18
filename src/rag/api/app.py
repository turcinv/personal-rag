"""FastAPI application + lifespan — the core value of the backend.

On startup the ``lifespan`` handler loads the embedding model, ChromaDB
collection, and cross-encoder reranker EXACTLY ONCE (via the cached getters in
:mod:`rag.query` — never reimplemented here) and stashes them on
``app.state.rag``. Routes read them through :func:`rag.api.deps.get_rag_state`,
so no request ever pays the model cold-start cost.

Entry point: ``rag-serve`` → :func:`run`. Also runnable as
``python -m rag.api.app`` (used by the compose ``command:``).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import query
from ..utils import load_config, setup_logging
from .routes import index as index_routes
from .routes import query as query_routes

logger = logging.getLogger("rag")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + collection + reranker once, share via ``app.state.rag``."""
    config = load_config()
    setup_logging(config)

    embedding_model = config.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model = config.get("reranker_model", query.DEFAULT_RERANKER)

    logger.info("API startup: loading embedding model %s", embedding_model)
    model = query.get_model(embedding_model)

    logger.info("API startup: opening collection")
    collection = query.open_collection(config)

    logger.info("API startup: loading reranker %s", reranker_model)
    reranker = query.get_reranker(reranker_model)

    count = collection.count()
    if count == 0:
        logger.warning(
            "API startup: collection %r is EMPTY (0 chunks). Serving anyway — an "
            "empty collection is a real state, not a crash. Run rag-index / "
            "POST /index to populate it.",
            collection.name,
        )
    else:
        logger.info("API startup: collection %r holds %d chunks", collection.name, count)

    app.state.rag = {
        "config": config,
        "model": model,
        "collection": collection,
        "reranker": reranker,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
    }

    logger.info("API startup complete.")
    yield
    logger.info("API shutdown.")


app = FastAPI(
    title="personal-rag API",
    description="HTTP backend for semantic retrieval over the vault + PDF library.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(query_routes.router)
app.include_router(index_routes.router)


def run() -> None:
    """``rag-serve`` entry point: start uvicorn on RAG_API_HOST:RAG_API_PORT."""
    import uvicorn

    host = os.environ.get("RAG_API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RAG_API_PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
