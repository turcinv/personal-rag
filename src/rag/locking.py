"""Cross-process writer lock keyed by the resolved index path.

Every process that mutates the vector store or a generation-bound sidecar — the
indexer CLI, the API's spawned reindex subprocess, the lexical build, backup, and
destructive collection maintenance — must hold this lock so two writers can never
touch one Chroma directory at once. The API keeps its own in-process
``threading.Lock`` as a fast local guard; this file-based lock is what serializes
*across* processes (CLI vs API vs a second shell).

Mechanism: an advisory ``flock`` on ``<index_path>.writer.lock``. ``flock`` is the
right primitive here because the kernel releases it automatically when the holding
process dies, so a crashed writer never leaves a permanently stuck lock — stale
locks self-heal. The lock file's *contents* (owner pid/host/operation/run id) are
purely informational: they let a blocked writer report who holds the lock, and let
readiness checks name the active writer.

Unix only (macOS dev + Jetson Linux prod); ``fcntl`` is unavailable on Windows,
which is not a target.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import socket
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


class IndexLockedError(RuntimeError):
    """Raised when another process already holds the index writer lock."""

    def __init__(self, path: Path, owner: "Optional[LockOwner]"):
        self.path = path
        self.owner = owner
        if owner is not None:
            detail = (
                f"held by {owner.operation} run {owner.run_id} "
                f"(pid {owner.pid} on {owner.host}, since {owner.acquired_at})"
            )
        else:
            detail = "held by another process"
        super().__init__(f"Index writer lock {path} is {detail}")


@dataclass(frozen=True)
class LockOwner:
    pid: int
    host: str
    operation: str
    run_id: str
    acquired_at: str

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, text: str) -> "Optional[LockOwner]":
        try:
            data = json.loads(text)
            return cls(
                pid=int(data["pid"]),
                host=str(data["host"]),
                operation=str(data["operation"]),
                run_id=str(data["run_id"]),
                acquired_at=str(data["acquired_at"]),
            )
        except (ValueError, KeyError, TypeError):
            return None


def lock_path_for(index_path) -> Path:
    """Return the lock-file path for a resolved index directory.

    The lock lives *beside* the Chroma directory (not inside it) so it never ends
    up in a backup copy or confuses Chroma's own directory scan.
    """
    resolved = Path(index_path).expanduser().resolve()
    return resolved.parent / f"{resolved.name}.writer.lock"


class IndexWriterLock:
    """Advisory exclusive lock over one index path, with owner reporting."""

    def __init__(
        self,
        index_path,
        operation: str,
        run_id: Optional[str] = None,
        timeout: float = 0.0,
        poll_interval: float = 0.25,
    ):
        self.path = lock_path_for(index_path)
        self.operation = operation
        self.run_id = run_id or uuid.uuid4().hex
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self._fd: Optional[int] = None
        self.reclaimed_from: Optional[LockOwner] = None

    def _read_owner(self) -> Optional[LockOwner]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        return LockOwner.from_json(text) if text.strip() else None

    def acquire(self) -> "IndexWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    os.close(fd)
                    raise
                if self.timeout <= 0 or time.monotonic() >= deadline:
                    owner = self._read_owner()
                    os.close(fd)
                    raise IndexLockedError(self.path, owner)
                time.sleep(self.poll_interval)

        # We hold the lock. Any pre-existing content is from a crashed writer
        # whose flock the kernel already released — record it, then overwrite.
        self.reclaimed_from = self._read_owner()
        owner = LockOwner(
            pid=os.getpid(),
            host=socket.gethostname(),
            operation=self.operation,
            run_id=self.run_id,
            acquired_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps(owner.as_dict()).encode("utf-8"))
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.ftruncate(self._fd, 0)
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "IndexWriterLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


def read_lock_owner(index_path) -> Optional[LockOwner]:
    """Return the current lock owner, or ``None`` if the lock is free.

    Non-destructive: probes with a non-blocking ``flock`` on a throwaway fd. If
    the probe acquires the lock the writer slot is free (returns ``None``); if it
    cannot, the lock is held and the recorded owner is returned. Used by API
    readiness reporting (Task 11) — never mutates the lock file.
    """
    path = lock_path_for(index_path)
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                return None
            return LockOwner.from_json(text) if text.strip() else None
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
            return None
    finally:
        os.close(fd)
