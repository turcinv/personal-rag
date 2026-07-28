"""Query surface: GET /health, POST /query, GET /status.

``GET /health`` is unauthenticated; ``POST /query`` and ``GET /status`` are
JWT-protected. Retrieval reuses ``rag.query`` (build_where + search) against the
model/store/reranker loaded once at startup — never reloaded per request.
"""

from fastapi import APIRouter, Depends

from ... import query as rag_query
from ..auth import require_jwt
from ..deps import get_rag_state
from ..schemas import HealthResponse, QueryRequest, QueryResponse, StatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — unauthenticated and cheap. Does not touch the model."""
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse, tags=["query"])
def query(
    request: QueryRequest,
    state: dict = Depends(get_rag_state),
    _claims: dict = Depends(require_jwt),
) -> QueryResponse:
    """Semantic retrieval over the once-loaded store.

    Maps ``filters`` → ``query.build_where`` (JSON ``type`` → the ``type_`` param;
    build_where returns None when everything is falsy) and calls ``query.search``
    with the model/store/config from app state. Returns search()'s native
    records inside a small envelope. ``reranked`` is True only when reranking was
    in effect AND actually applied (a non-empty dense pool produced scores).
    Omitting ``rerank`` in the request uses the profile's ``rerank_default``."""
    f = request.filters
    where = rag_query.build_where(
        domain=f.domain if f else None,
        type_=f.type if f else None,
        source=f.source if f else None,
        confidence=f.confidence if f else None,
        subdomain=f.subdomain if f else None,
        status=f.status if f else None,
    )

    # rerank omitted (None) → fall back to the profile's rerank_default.
    rerank = (
        request.rerank
        if request.rerank is not None
        else rag_query.rerank_default(state["config"])
    )

    records = rag_query.search(
        request.query,
        n_results=request.n_results,
        filters=where,
        tags=f.tags if f else None,
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
    state: dict = Depends(get_rag_state), _claims: dict = Depends(require_jwt)
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
