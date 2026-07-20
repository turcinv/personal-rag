"""Query surface: GET /health, POST /query, GET /status.

``GET /health`` is unauthenticated; ``POST /query`` and ``GET /status`` are
JWT-protected. Retrieval reuses ``rag.query`` (build_where + search) against the
model/collection/reranker loaded once at startup — never reloaded per request.
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
    """Semantic retrieval over the once-loaded collection.

    Maps ``filters`` → ``query.build_where`` (JSON ``type`` → the ``type_`` param;
    build_where returns None when everything is falsy) and calls ``query.search``
    with the model/collection/config from app state. Returns search()'s native
    records inside a small envelope. ``reranked`` is True only when reranking was
    requested AND actually applied (a non-empty dense pool produced scores)."""
    f = request.filters
    where = rag_query.build_where(
        domain=f.domain if f else None,
        type_=f.type if f else None,
        source=f.source if f else None,
        confidence=f.confidence if f else None,
        subdomain=f.subdomain if f else None,
        status=f.status if f else None,
    )

    records = rag_query.search(
        request.query,
        n_results=request.n_results,
        filters=where,
        tags=f.tags if f else None,
        config=state["config"],
        model=state["model"],
        collection=state["collection"],
        rerank=request.rerank,
    )

    reranked = request.rerank and any("rerank_score" in r for r in records)
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
    """Report live ChromaDB state: is the index actually populated? Reads the
    once-loaded collection from app state and counts chunks."""
    collection = state["collection"]
    return StatusResponse(
        collection=collection.name,
        count=collection.count(),
        embedding_model=state["embedding_model"],
        reranker_model=state["reranker_model"],
    )
