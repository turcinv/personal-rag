"""Query surface: GET /health, POST /query, GET /status.

``GET /health`` is unauthenticated; ``GET /status`` reports live Chroma state and
is JWT-protected. ``POST /query`` is a stub filled in by Phase 3.
"""

from fastapi import APIRouter, Depends

from ..auth import require_jwt
from ..deps import get_rag_state
from ..schemas import HealthResponse, StatusResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — unauthenticated and cheap. Does not touch the model."""
    return HealthResponse(status="ok")


@router.post("/query", tags=["query"])
def query(state: dict = Depends(get_rag_state), _claims: dict = Depends(require_jwt)):
    """TODO(Phase 3): map filters via ``query.build_where`` and call
    ``query.search`` with the once-loaded model/collection/reranker."""
    raise NotImplementedError("POST /query is implemented in Phase 3")


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
