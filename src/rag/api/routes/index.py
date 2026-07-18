"""Indexing surface: POST /index, GET /index/jobs/{job_id}.

``POST /index`` launches ``python -m rag.indexer`` out of the request path as a
subprocess (see :mod:`rag.api.jobs`) and returns 202 immediately with a job_id;
it refuses a second run with 409 while one is active. ``GET /index/jobs/{job_id}``
reports the tracked status. Both are JWT-protected. The response never exposes
server-side log paths; a failed run (including the indexer's 0-files anti-wipe
guard) surfaces as ``status="failed"`` with a short ``error``.
"""

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import require_jwt
from ..deps import get_rag_state
from ..jobs import manager
from ..schemas import IndexJobResponse

router = APIRouter(prefix="/index", tags=["index"])


def _log_dir(state: dict) -> Path:
    """Resolve the directory for indexer job logs from the same config/env the
    logger uses (RAG_LOG_PATH → config log_path → ./logs/rag.log)."""
    log_path = (
        os.environ.get("RAG_LOG_PATH")
        or state["config"].get("log_path")
        or "logs/rag.log"
    )
    return Path(log_path).expanduser().parent


@router.post("", response_model=IndexJobResponse, status_code=status.HTTP_202_ACCEPTED)
def start_index(
    state: dict = Depends(get_rag_state),
    _claims: dict = Depends(require_jwt),
) -> IndexJobResponse:
    """Kick off a reindex as a background subprocess. 202 + job_id on success;
    409 if an index run is already in progress (never two against one Chroma
    dir). The command is fixed — no request input is passed to the subprocess."""
    job_id, record = manager.start(_log_dir(state))
    if job_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="an index run is already in progress",
        )
    return IndexJobResponse(job_id=job_id, status=record["status"])


@router.get("/jobs/{job_id}", response_model=IndexJobResponse)
def job_status(job_id: str, _claims: dict = Depends(require_jwt)) -> IndexJobResponse:
    """Return the tracked status for ``job_id`` (404 if unknown). No server-side
    log paths are included in the body."""
    record = manager.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="unknown job_id",
        )
    return IndexJobResponse(
        job_id=job_id,
        status=record["status"],
        started=record["started"],
        finished=record["finished"],
        returncode=record["returncode"],
        error=record["error"],
    )
