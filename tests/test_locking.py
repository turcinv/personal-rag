"""Offline tests for the cross-process index writer lock.

Uses two lock objects over one temp index path in a single process — flock
associates the lock with the open file description, so two independent
open()+flock calls contend exactly as two processes would. No subprocess, no
Chroma, no network.
"""

import json

import pytest

from rag.locking import (
    IndexLockedError,
    IndexWriterLock,
    LockOwner,
    lock_path_for,
    read_lock_owner,
)


def test_lock_file_is_a_sibling_of_the_index_dir(tmp_path):
    index_path = tmp_path / "chroma_db"
    assert lock_path_for(index_path) == tmp_path / "chroma_db.writer.lock"


def test_second_writer_is_rejected_while_lock_is_held(tmp_path):
    index_path = tmp_path / "chroma_db"
    first = IndexWriterLock(index_path, "rag-index", run_id="run-a")
    first.acquire()
    try:
        second = IndexWriterLock(index_path, "rag-build-lexical", run_id="run-b")
        with pytest.raises(IndexLockedError) as excinfo:
            second.acquire()
        owner = excinfo.value.owner
        assert owner is not None
        assert owner.operation == "rag-index"
        assert owner.run_id == "run-a"
    finally:
        first.release()


def test_lock_is_reacquirable_after_release(tmp_path):
    index_path = tmp_path / "chroma_db"
    with IndexWriterLock(index_path, "rag-index"):
        pass
    # A second writer succeeds once the first released.
    with IndexWriterLock(index_path, "rag-backup") as lock:
        assert lock.reclaimed_from is None


def test_stale_owner_content_is_reclaimed_safely(tmp_path):
    index_path = tmp_path / "chroma_db"
    lock_file = lock_path_for(index_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    # Simulate a crashed writer: owner JSON on disk, but no process holds flock
    # (the kernel released it on death), so a new writer must reclaim it.
    stale = LockOwner(
        pid=999999, host="dead-host", operation="rag-index",
        run_id="crashed", acquired_at="2026-01-01T00:00:00",
    )
    lock_file.write_text(json.dumps(stale.as_dict()), encoding="utf-8")

    with IndexWriterLock(index_path, "rag-index") as lock:
        assert lock.reclaimed_from is not None
        assert lock.reclaimed_from.run_id == "crashed"


def test_read_lock_owner_reports_free_and_held(tmp_path):
    index_path = tmp_path / "chroma_db"
    assert read_lock_owner(index_path) is None  # no lock file yet

    with IndexWriterLock(index_path, "rag-index", run_id="held"):
        owner = read_lock_owner(index_path)
        assert owner is not None
        assert owner.run_id == "held"

    # After release the slot is free again.
    assert read_lock_owner(index_path) is None
