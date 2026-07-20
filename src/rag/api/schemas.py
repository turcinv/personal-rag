"""Pydantic request/response models for the HTTP backend.

Health/status (Phase 2), query (Phase 3), and index-job (Phase 4 — stub) models.
"""

from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field, StringConstraints


class HealthResponse(BaseModel):
    """Liveness probe payload (unauthenticated ``GET /health``)."""

    status: str = "ok"


class StatusResponse(BaseModel):
    """Index status: is the collection actually populated? (``GET /status``)."""

    collection: str
    count: int
    embedding_model: str
    reranker_model: str


# ── Query (Phase 3) ────────────────────────────────────────────────────────────


class QueryFilters(BaseModel):
    """Optional metadata constraints mapped to ``query.build_where``. The JSON
    field ``type`` maps to build_where's ``type_`` parameter in the handler."""

    domain: Optional[str] = None
    subdomain: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[str] = None


class QueryRequest(BaseModel):
    """Semantic query request. ``n_results`` is bounded 1..50 to protect the
    Jetson's 8 GB memory budget; empty/whitespace queries are rejected (422)."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    n_results: int = Field(8, ge=1, le=50)
    rerank: bool = True
    filters: Optional[QueryFilters] = None


class QueryResponse(BaseModel):
    """Envelope around ``query.search``'s native record shape."""

    query: str
    count: int
    reranked: bool
    results: list[dict[str, Any]]


# ── Indexing (Phase 4) ─────────────────────────────────────────────────────────


class IndexJobResponse(BaseModel):
    """Returned by ``POST /index`` (202) and ``GET /index/jobs/{job_id}``.

    Deliberately omits the server-side log path to avoid host-path leakage; on
    failure ``error`` carries a short reason (e.g. the indexer's tail log)."""

    job_id: str
    status: str
    started: Optional[str] = None
    finished: Optional[str] = None
    returncode: Optional[int] = None
    error: Optional[str] = None
