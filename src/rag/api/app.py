"""FastAPI application + lifespan — the core value of the backend.

On startup the ``lifespan`` handler loads the embedding model, opens the
retrieval store (see :mod:`rag.store`), and loads the cross-encoder reranker
EXACTLY ONCE (via the cached getters in :mod:`rag.query` — never reimplemented
here) and stashes them on ``app.state.rag``. Routes read them through
:func:`rag.api.deps.get_rag_state`, so no request ever pays the model
cold-start cost.

Entry point: ``rag-serve`` → :func:`run`. Also runnable as
``python -m rag.api.app`` (used by the compose ``command:``).
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .. import query
from ..generation import GenerationConfigError, get_generator
from ..utils import load_config, setup_logging
from .routes import answer as answer_routes
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

    logger.info("API startup: opening store")
    store = query.open_store(config)

    logger.info("API startup: loading reranker %s", reranker_model)
    reranker = query.get_reranker(reranker_model)

    count = store.count()
    if count == 0:
        logger.warning(
            "API startup: collection %r is EMPTY (0 chunks). Serving anyway — an "
            "empty collection is a real state, not a crash. Run rag-index / "
            "POST /index to populate it.",
            store.name,
        )
    else:
        logger.info("API startup: collection %r holds %d chunks", store.name, count)

    # Answer generation is optional. Build the generator once here; if it is not
    # configured (no `generation` block or no API key), keep it None so /query
    # still works and /answer returns 503. An unknown provider is a real config
    # bug and is left to raise loudly.
    generator = None
    try:
        generator = get_generator(config)
        logger.info(
            "API startup: generation enabled (%s / %s)",
            generator.provider,
            generator.model,
        )
    except GenerationConfigError as exc:
        logger.info("API startup: generation disabled — %s", exc)

    app.state.rag = {
        "config": config,
        "model": model,
        "store": store,
        "reranker": reranker,
        "generator": generator,
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
app.include_router(answer_routes.router)
app.include_router(index_routes.router)


def run() -> None:
    """``rag-serve`` entry point: start uvicorn on RAG_API_HOST:RAG_API_PORT."""
    import uvicorn

    host = os.environ.get("RAG_API_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RAG_API_PORT", DEFAULT_PORT))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
