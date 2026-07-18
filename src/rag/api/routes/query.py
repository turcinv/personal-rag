"""Query surface: GET /health, POST /query, GET /status.

Phase 2: only ``GET /health`` (unauthenticated) is live so the app is verifiably
alive. ``POST /query`` and ``GET /status`` are stubs filled in by Phases 3.
"""

from fastapi import APIRouter, Depends

from ..auth import require_jwt
from ..deps import get_rag_state
from ..schemas import HealthResponse

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


@router.get("/status", tags=["ops"])
def status(state: dict = Depends(get_rag_state), _claims: dict = Depends(require_jwt)):
    """TODO(Phase 3): report collection name, chunk count, embedding + reranker
    models from the shared app state."""
    raise NotImplementedError("GET /status is implemented in Phase 3")
