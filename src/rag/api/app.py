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
from ..generation import GenerationDisabledError, get_generator
from ..utils import apply_offline_mode, load_config, setup_logging
from .auth import (
    JWT_AUDIENCE_ENV,
    JWT_ISSUER_ENV,
    JWT_SECRET_ENV,
    validate_secret_strength,
)
from .concurrency import InferenceLimiter
from .routes import answer as answer_routes
from .routes import index as index_routes
from .routes import query as query_routes

logger = logging.getLogger("rag")

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _export_jwt_claims_env(config) -> None:
    """Propagate configured JWT audience/issuer into the env require_jwt reads.

    Keeps a single source of truth (the config's ``api`` block) while letting the
    stateless ``require_jwt`` dependency enforce aud/iss without app-state access.
    An explicit environment override is left untouched.
    """
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    if api.get("jwt_audience") and not os.environ.get(JWT_AUDIENCE_ENV):
        os.environ[JWT_AUDIENCE_ENV] = str(api["jwt_audience"])
    if api.get("jwt_issuer") and not os.environ.get(JWT_ISSUER_ENV):
        os.environ[JWT_ISSUER_ENV] = str(api["jwt_issuer"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + collection + reranker once, share via ``app.state.rag``."""
    config = load_config()
    setup_logging(config)

    # Fail loudly at startup on a weak secret rather than accepting brute-forceable
    # tokens per request. Also propagate configured aud/iss so require_jwt enforces
    # them (it reads these env vars at request time).
    validate_secret_strength(os.environ.get(JWT_SECRET_ENV))
    _export_jwt_claims_env(config)

    # Enforced offline mode (HF_HUB_OFFLINE etc.) must be set before any model
    # load so a cold cache fails fast instead of reaching the network.
    apply_offline_mode(config)

    embedding_model = config.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model = config.get("reranker_model", query.DEFAULT_RERANKER)
    embedding_revision = config.get("embedding_revision") or None

    logger.info("API startup: loading embedding model %s", embedding_model)
    model = query.get_model(embedding_model, revision=embedding_revision)

    logger.info("API startup: opening store")
    store = query.open_store(config, model=model)

    # Load the cross-encoder only when the profile actually reranks by default.
    # The personal profile sets rerank_default=False, so on the Jetson the
    # reranker is never resident unless a request explicitly forces rerank=True
    # (search() then lazily loads it). Saves memory and startup time.
    reranker = None
    if query.rerank_default(config):
        logger.info("API startup: loading reranker %s", reranker_model)
        reranker = query.get_reranker(reranker_model)
    else:
        logger.info(
            "API startup: reranker %s not preloaded (rerank_default=False)",
            reranker_model,
        )

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

    # Answer generation is optional. Only a *disabled* state (no generation block
    # or no API key) is swallowed to keep /query working while /answer returns
    # 503. An *invalid* config (unknown provider, insecure base_url, missing
    # model) raises GenerationConfigError and is NOT caught — the server fails
    # loudly at startup rather than silently degrading.
    generator = None
    try:
        generator = get_generator(config)
        logger.info(
            "API startup: generation enabled (%s / %s)",
            generator.provider,
            generator.model,
        )
    except GenerationDisabledError as exc:
        logger.info("API startup: generation disabled — %s", exc)

    inference_limiter = InferenceLimiter(
        max_concurrency=int(config.get("api", {}).get("inference_concurrency", 1))
        if isinstance(config.get("api"), dict)
        else 1
    )

    app.state.rag = {
        "config": config,
        "model": model,
        "store": store,
        "reranker": reranker,
        "generator": generator,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "inference_limiter": inference_limiter,
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
    """``rag-serve`` entry point. Host/port precedence: env > config.api > default."""
    import uvicorn

    api = {}
    try:
        cfg = load_config()
        api = cfg.get("api") if isinstance(cfg.get("api"), dict) else {}
    except Exception as exc:  # pragma: no cover - config errors surface in lifespan
        logger.warning("Could not read api host/port from config: %s", exc)

    host = os.environ.get("RAG_API_HOST") or api.get("host") or DEFAULT_HOST
    port = int(os.environ.get("RAG_API_PORT") or api.get("port") or DEFAULT_PORT)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
