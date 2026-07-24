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
    field ``type`` maps to build_where's ``type_`` parameter in the handler.

    ``status`` is a native ``$eq`` clause (build_where); ``tags`` is a post-filter
    applied inside ``query.search`` (exact, case-insensitive membership; multiple
    tags = AND) — Chroma can't filter the comma-joined tags string natively."""

    domain: Optional[str] = None
    subdomain: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[str] = None
    status: Optional[str] = None
    tags: Optional[list[str]] = None


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


# ── Answer generation (retrieval ≠ chatbot) ─────────────────────────────────────


class AnswerRequest(BaseModel):
    """Ask a question and get a grounded, cited answer synthesized from the
    retrieved chunks. Retrieval knobs mirror ``QueryRequest`` (same
    ``build_where`` + ``search`` path); ``max_tokens`` / ``temperature`` override
    the configured generation defaults for this one call.

    ``n_results`` is the number of chunks fed to the LLM as context — capped at
    20 (not 50 like /query) to keep the prompt and cost bounded."""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    n_results: int = Field(8, ge=1, le=20)
    rerank: bool = True
    filters: Optional[QueryFilters] = None
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)


class Citation(BaseModel):
    """One numbered source passage behind an answer. ``n`` matches the ``[n]``
    markers the model cites; the fields identify the source note/chunk."""

    n: int
    title: Optional[str] = None
    path: Optional[str] = None
    domain: Optional[str] = None
    distance: Optional[float] = None
    rerank_score: Optional[float] = None


class AnswerResponse(BaseModel):
    """Envelope for a generated answer.

    ``grounded`` is False when retrieval returned no chunks — the endpoint then
    returns a fixed "no relevant context" answer WITHOUT calling the LLM, so it
    never hallucinates from an empty context. ``citations`` line up 1:1 with the
    context passages the model was given."""

    query: str
    answer: str
    grounded: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    reranked: bool = False
    citations: list[Citation] = []
    usage: Optional[dict[str, Any]] = None


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
