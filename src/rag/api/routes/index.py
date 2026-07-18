"""Indexing surface: POST /index, GET /index/jobs/{job_id}.

Phase 2 stubs. Phase 4 launches ``python -m rag.indexer`` out of the request
path (subprocess), tracks jobs in an in-process registry, returns 202 + job_id,
and refuses a second concurrent run with 409. The indexer's 0-files anti-wipe
guard is preserved and surfaced as a failed job.
"""

from fastapi import APIRouter, Depends

from ..auth import require_jwt
from ..deps import get_rag_state

router = APIRouter(prefix="/index", tags=["index"])


@router.post("")
def start_index(state: dict = Depends(get_rag_state), _claims: dict = Depends(require_jwt)):
    """TODO(Phase 4): launch a reindex subprocess, register the job, return 202
    with a job_id. Refuse (409) if a run is already in progress."""
    raise NotImplementedError("POST /index is implemented in Phase 4")


@router.get("/jobs/{job_id}")
def job_status(job_id: str, _claims: dict = Depends(require_jwt)):
    """TODO(Phase 4): return the tracked status for ``job_id``."""
    raise NotImplementedError("GET /index/jobs/{job_id} is implemented in Phase 4")
