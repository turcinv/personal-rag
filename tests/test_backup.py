"""Offline tests for quiesced backup + verified restore.

Builds a real ChromaStore in a temp dir (like tests/test_store.py), stamps
provenance, backs it up, restores to another temp path, and verifies counts,
provenance, and a model-free representative query. No model, no network, no
touch of the real corpus/index.
"""

import pytest

from rag.backup import (
    BackupError,
    create_backup,
    restore_backup,
    restore_drill,
    verify_backup,
    verify_restore,
)
from rag.locking import IndexWriterLock
from rag.provenance import IndexProvenance, begin_generation
from rag.store.chroma_store import ChromaStore


def _provenance():
    return IndexProvenance(
        embedding_model="test-model",
        embedding_revision="",
        embedding_dimension=4,
        normalized=True,
        chunker_version="heading-paragraph-v1",
        chunk_max_chars=1200,
        chunk_overlap_chars=150,
        metric="cosine",
        corpus_profile="test",
    )


def _seed_index(index_path, collection="backup-test"):
    store = ChromaStore(str(index_path), collection)
    store.ensure(collection)
    store.upsert(
        ids=["a", "b", "c"],
        embeddings=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]],
        docs=["alpha", "bravo", "charlie"],
        metas=[{"path": "p1"}, {"path": "p2"}, {"path": "p3"}],
    )
    begin_generation(store, _provenance())
    return collection


def _config(index_path, collection):
    return {
        "store": "chroma",
        "index_path": str(index_path),
        "collection_name": collection,
        "embedding_dimension": 4,
    }


def test_create_verify_restore_roundtrip(tmp_path):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)
    config = _config(index_path, collection)

    backup_dir = create_backup(config, tmp_path / "backup", collection)
    assert verify_backup(backup_dir)["chunk_count"] == 3

    dest = tmp_path / "restored"
    restore_backup(backup_dir, dest)
    report = verify_restore(backup_dir, dest)
    assert report["chunk_count"] == 3
    assert report["provenance_verified"] is True
    assert report["representative_hits"] == 3


def test_restore_drill_passes_on_temp_path(tmp_path):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)
    backup_dir = create_backup(_config(index_path, collection), tmp_path / "backup", collection)

    report = restore_drill(backup_dir)
    assert report["drill"] is True
    assert report["chunk_count"] == 3


def test_verify_detects_checksum_tampering(tmp_path):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)
    backup_dir = create_backup(_config(index_path, collection), tmp_path / "backup", collection)

    sqlite_file = backup_dir / "chroma" / "chroma.sqlite3"
    with sqlite_file.open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(BackupError, match="checksum mismatch"):
        verify_backup(backup_dir)


def test_backup_refuses_nonempty_destination(tmp_path):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)
    dest = tmp_path / "backup"
    dest.mkdir()
    (dest / "stale").write_text("x", encoding="utf-8")

    with pytest.raises(BackupError, match="not empty"):
        create_backup(_config(index_path, collection), dest, collection)


def test_restore_refuses_nonempty_destination(tmp_path):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)
    backup_dir = create_backup(_config(index_path, collection), tmp_path / "backup", collection)

    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "stale").write_text("x", encoding="utf-8")
    with pytest.raises(BackupError, match="not empty"):
        restore_backup(backup_dir, dest)


def test_backup_holds_writer_lock_during_copy(tmp_path, monkeypatch):
    index_path = tmp_path / "chroma_db"
    collection = _seed_index(index_path)

    # Prove the backup acquired the index writer lock: while a competing writer
    # holds it, create_backup must fail fast rather than copy a live database.
    holder = IndexWriterLock(index_path, "rag-index")
    holder.acquire()
    try:
        from rag.locking import IndexLockedError

        with pytest.raises(IndexLockedError):
            create_backup(_config(index_path, collection), tmp_path / "backup", collection)
    finally:
        holder.release()
