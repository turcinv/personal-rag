"""Query surface: GET /health, POST /query, GET /status.

``GET /health`` is unauthenticated; ``POST /query`` and ``GET /status`` are
JWT-protected. Retrieval reuses ``rag.query`` (build_where + search) against the
model/store/reranker loaded once at startup — never reloaded per request.
"""

from fastapi import APIRouter, Depends

from ... import query as rag_query
from ...retrieval import RetrievalFilter
from ..auth import SCOPE_QUERY, require_scope
from ..concurrency import guard
from ..deps import get_rag_state
from ..schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    ReadyResponse,
    StatusResponse,
)
from ...locking import read_lock_owner
from ...provenance import read_index_state

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — unauthenticated and cheap. Does not touch the model."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready(state: dict = Depends(get_rag_state)):
    """Readiness probe — unauthenticated. Reports whether the server can serve.

    ``ready`` is True only when the model and a populated store are both loaded.
    Also surfaces the active vector generation, whether answer generation is
    wired, and any current index-writer-lock owner — enough for an operator or
    orchestrator to see the real serving state without a token. Returns 503 when
    not ready so a load balancer holds traffic until the index is populated.
    """
    from fastapi import Response
    from fastapi import status as http_status

    store = state.get("store")
    model = state.get("model")
    count = None
    generation_id = None
    if store is not None:
        try:
            count = int(store.count())
        except Exception:  # pragma: no cover - defensive; a broken store isn't ready
            count = None
        index_state = read_index_state(store)
        generation_id = index_state.generation_id if index_state is not None else None

    lock_owner = None
    config = state.get("config") or {}
    try:
        owner = read_lock_owner(config.get("index_path", "./chroma_db"))
        if owner is not None:
            lock_owner = f"{owner.operation}:{owner.run_id}"
    except Exception:  # pragma: no cover - lock probing must never break readiness
        lock_owner = None

    is_ready = model is not None and count is not None and count > 0
    body = ReadyResponse(
        ready=is_ready,
        store_populated=bool(count),
        count=count,
        generation_id=generation_id,
        generation_enabled=state.get("generator") is not None,
        index_locked_by=lock_owner,
    )
    if not is_ready:
        return Response(
            content=body.model_dump_json(),
            media_type="application/json",
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return body


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(
    request: QueryRequest,
    state: dict = Depends(get_rag_state),
    _claims: dict = Depends(require_scope(SCOPE_QUERY)),
) -> QueryResponse:
    """Semantic retrieval over the once-loaded store.

    Maps ``filters`` → a backend-neutral ``RetrievalFilter`` (the store compiles
    it; the route constructs no Chroma syntax) and calls ``query.search`` with
    the model/store/config from app state. Returns search()'s native
    records inside a small envelope. ``reranked`` is True only when reranking was
    in effect AND actually applied (a non-empty dense pool produced scores).
    Omitting ``rerank`` in the request uses the profile's ``rerank_default``."""
    retrieval_filter = RetrievalFilter.from_api_filters(request.filters)

    # rerank omitted (None) → fall back to the profile's rerank_default.
    rerank = (
        request.rerank
        if request.rerank is not None
        else rag_query.rerank_default(state["config"])
    )

    # Hybrid retrieval is intentionally a CLI/eval-only surface (needs a built
    # lexical sidecar); the HTTP API stays dense-only for predictable latency.
    # The inference slot bounds concurrent embed/rerank work (Jetson memory).
    with guard(state):
        records = rag_query.search(
            request.query,
            n_results=request.n_results,
            retrieval_filter=retrieval_filter,
            config=state["config"],
            model=state["model"],
            store=state["store"],
            rerank=rerank,
        )

    reranked = rerank and any("rerank_score" in r for r in records)
    return QueryResponse(
        query=request.query,
        count=len(records),
        reranked=reranked,
        results=records,
    )


@router.get("/status", response_model=StatusResponse, tags=["ops"])
def status(
    state: dict = Depends(get_rag_state),
    _claims: dict = Depends(require_scope(SCOPE_QUERY)),
) -> StatusResponse:
    """Report live store state: is the index actually populated? Reads the
    once-loaded store from app state and counts chunks."""
    store = state["store"]
    return StatusResponse(
        collection=store.name,
        count=store.count(),
        embedding_model=state["embedding_model"],
        reranker_model=state["reranker_model"],
    )
