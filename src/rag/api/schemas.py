"""Pydantic request/response models for the HTTP backend.

Phase 2: HealthResponse and StatusResponse are defined here. QueryRequest /
QueryResponse and the index-job schemas are stubs to be fleshed out in Phases
3-4; keep them importable so routes can reference them.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness probe payload (unauthenticated ``GET /health``)."""

    status: str = "ok"


class StatusResponse(BaseModel):
    """Index status: is the collection actually populated? (``GET /status``)."""

    collection: str
    count: int
    embedding_model: str
    reranker_model: str


# ── Query (Phase 3 — stubs) ────────────────────────────────────────────────────


class QueryFilters(BaseModel):
    """Optional metadata constraints mapped to ``query.build_where``."""

    domain: Optional[str] = None
    subdomain: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[str] = None


class QueryRequest(BaseModel):
    """Semantic query request. ``n_results`` is bounded 1..50 to protect the
    Jetson's 8 GB memory budget."""

    query: str = Field(..., min_length=1)
    n_results: int = Field(8, ge=1, le=50)
    rerank: bool = True
    filters: Optional[QueryFilters] = None


class QueryResponse(BaseModel):
    """Envelope around ``query.search``'s native record shape."""

    query: str
    count: int
    reranked: bool
    results: list[dict[str, Any]]


# ── Indexing (Phase 4 — stubs) ─────────────────────────────────────────────────


class IndexJobResponse(BaseModel):
    """Returned by ``POST /index`` (202) and ``GET /index/jobs/{job_id}``."""

    job_id: str
    status: str
    started: Optional[str] = None
    finished: Optional[str] = None
    returncode: Optional[int] = None
