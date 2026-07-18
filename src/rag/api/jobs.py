"""In-process background-job manager for reindex runs.

``POST /index`` must never run the (minutes-long) indexer in the request path or
block the event loop, and two indexers must never run against the same Chroma
dir at once. So this module launches ``python -m rag.indexer`` as a *subprocess*
(fixed argv — no shell, no user input), tracks it in a module-level registry
guarded by a lock, and watches it from a daemon thread that flips the job record
to ``succeeded``/``failed`` when the process exits.

The subprocess inherits the current environment, so it hits the SAME Chroma dir
(via RAG_INDEX_PATH etc.) and the SAME 0-files anti-wipe guard in
``indexer.main()`` — that guard is not reimplemented or bypassed here; a run it
aborts simply exits nonzero and surfaces as a ``failed`` job whose ``error`` is
the tail of the indexer log (the RuntimeError message).

Test seam: monkeypatch :func:`_spawn_indexer` to return a fake process object
(anything with ``.wait()`` and ``.returncode``) — no real subprocess, no real
Chroma writes. ``JobManager.join`` lets a test await the monitor thread and
observe the resulting status transition.
"""

import logging
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("rag")

INDEXER_MODULE = "rag.indexer"

# Statuses that mean an indexer is (about to be) running — used by the
# concurrency guard to refuse a second run.
ACTIVE_STATUSES = ("queued", "running")


def _now() -> str:
    """UTC timestamp (ISO 8601) for job start/finish records."""
    return datetime.now(timezone.utc).isoformat()


def _tail_log(log_path: str, max_chars: int = 500) -> str:
    """Return the tail of the indexer log as a short failure reason.

    For the 0-files anti-wipe guard this is the RuntimeError message. Never
    raises: an unreadable/absent log yields a generic message."""
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "indexer exited nonzero (no log available)"
    if not text:
        return "indexer exited nonzero (empty log)"
    return text[-max_chars:].strip()


def _spawn_indexer(log_path: Path) -> subprocess.Popen:
    """Launch the indexer as a subprocess writing to ``log_path``.

    MONKEYPATCH SEAM: tests replace this function to return a fake process with
    ``.wait()`` / ``.returncode`` so no real indexer runs and no real Chroma dir
    is touched. Fixed argv list; never ``shell=True``; no user input is ever
    interpolated. Inherits the current environment so RAG_* paths carry over."""
    log_file = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", INDEXER_MODULE],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    finally:
        # The child has its own dup'd fd; the parent's handle is no longer needed.
        log_file.close()
    return proc


class JobManager:
    """Tracks reindex jobs in a lock-guarded, single-process registry."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def start(self, log_dir: Path) -> tuple[str | None, dict | None]:
        """Start a reindex job. Returns ``(job_id, record)`` or ``(None, None)``
        if one is already active (the caller then returns HTTP 409).

        The active-check and record insertion happen together under the lock so
        two near-simultaneous POSTs can't both pass the guard. The subprocess is
        spawned (non-blocking) while still holding the lock, so a second POST
        arriving mid-spawn already sees a ``running`` job."""
        with self._lock:
            if any(j["status"] in ACTIVE_STATUSES for j in self._jobs.values()):
                return None, None

            job_id = uuid.uuid4().hex
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_path = Path(log_dir) / f"index-{job_id}.log"
            record = {
                "status": "running",
                "started": _now(),
                "finished": None,
                "returncode": None,
                "log_path": str(log_path),
                "error": None,
            }
            self._jobs[job_id] = record

            try:
                proc = _spawn_indexer(log_path)
            except Exception as exc:  # launch failure — record and report
                record["status"] = "failed"
                record["finished"] = _now()
                record["error"] = f"failed to launch indexer: {exc}"
                logger.error("index job %s failed to launch: %s", job_id, exc)
                return job_id, dict(record)

            thread = threading.Thread(
                target=self._monitor, args=(job_id, proc), daemon=True,
                name=f"index-monitor-{job_id}",
            )
            self._threads[job_id] = thread
            thread.start()
            logger.info("index job %s started (pid=%s)", job_id, getattr(proc, "pid", "?"))
            return job_id, dict(record)

    def get(self, job_id: str) -> dict | None:
        """Return a copy of the job record, or None if unknown."""
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    def join(self, job_id: str, timeout: float | None = None) -> None:
        """Wait for a job's monitor thread to finish (test/introspection helper)."""
        thread = self._threads.get(job_id)
        if thread is not None:
            thread.join(timeout)

    # ── internals ───────────────────────────────────────────────────────────

    def _monitor(self, job_id: str, proc) -> None:
        """Block on the subprocess (off the event loop), then update the record."""
        returncode = proc.wait()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record["returncode"] = returncode
            record["finished"] = _now()
            if returncode == 0:
                record["status"] = "succeeded"
                logger.info("index job %s succeeded", job_id)
            else:
                record["status"] = "failed"
                record["error"] = _tail_log(record["log_path"])
                logger.warning("index job %s failed (rc=%s)", job_id, returncode)


# Module-level singleton — single process, so one registry is authoritative.
manager = JobManager()
