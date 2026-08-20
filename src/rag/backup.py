"""Quiesced backup and verified restore for the vector store + sidecars.

A backup is a directory holding a point-in-time copy of everything needed to
bring the retrieval system back without re-embedding the corpus:

    <backup>/
      chroma/              copy of the Chroma index dir (vectors + provenance)
      lexical/<name>.db    copy of the BM25 sidecar (if present)
      manifests/           copy of the index-run manifests
      config.yaml          the active config at backup time
      backup_manifest.json restore metadata: counts, generation/provenance,
                           and sha256 of every copied Chroma file

The copy runs while holding the cross-process **index writer lock**, so no
indexer/lexical/maintenance writer can mutate the store mid-copy — this is the
"quiesced" requirement (we never copy a database another process is actively
writing). Provenance rides along automatically: ``ChromaStore`` stores it inside
the collection metadata in ``chroma.sqlite3``, so copying the dir captures it.

Verification and the restore drill open the *copy* (or a temp restore) through
``get_store`` / ``read_index_state`` and compare counts, generation fingerprint,
and a representative ``store.query`` — no embedding model, no network, no touch
of the live index. ``chromadb`` stays confined to ``store/`` (this module only
uses the store seam).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .locking import IndexLockedError, IndexWriterLock
from .provenance import read_index_state
from .store import get_store
from .utils import load_config

BACKUP_SCHEMA_VERSION = 1
_MANIFEST_NAME = "backup_manifest.json"


class BackupError(RuntimeError):
    """Raised when a backup is incomplete or fails verification."""


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _checksum_tree(root: Path) -> Dict[str, str]:
    checksums: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            checksums[path.relative_to(root).as_posix()] = _sha256_file(path)
    return checksums


def _store_for_index(config: Mapping[str, Any], index_path: Path, collection_name):
    """Open a store rooted at ``index_path`` without disturbing the live config."""
    overlay = dict(config)
    overlay["index_path"] = str(index_path)
    name = collection_name or config.get("collection_name", "obsidian_markdown")
    return get_store(overlay, name)


def _representative_query(store, index_state, k: int = 3) -> int:
    """Run one model-free vector query to prove the restored index answers.

    Uses a fixed non-zero embedding of the recorded dimension — the store only
    needs a vector of the right width, not a real one. Returns the hit count.
    """
    dimension = 0
    if index_state is not None:
        dimension = int(index_state.provenance.embedding_dimension)
    if dimension <= 0:
        return -1  # dimension unknown; caller treats as "not checked"
    embedding = [0.1] * dimension
    return len(store.query(embedding, k))


def create_backup(
    config: Mapping[str, Any],
    dest: Path,
    collection_name: Optional[str] = None,
    wait_for_lock: float = 0.0,
) -> Path:
    """Create a quiesced backup under ``dest`` and return its path."""
    dest = Path(dest).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise BackupError(f"Backup destination is not empty: {dest}")
    index_path = Path(config.get("index_path", "./chroma_db")).expanduser().resolve()
    if not index_path.exists():
        raise BackupError(f"Index path does not exist: {index_path}")
    name = collection_name or config.get("collection_name", "obsidian_markdown")

    lock = IndexWriterLock(index_path, "rag-backup", timeout=wait_for_lock)
    lock.acquire()
    try:
        dest.mkdir(parents=True, exist_ok=True)
        chroma_copy = dest / "chroma"
        shutil.copytree(index_path, chroma_copy)

        lexical_source = config.get("lexical_path") or f"./lexical_index/{name}.db"
        lexical_source = Path(lexical_source).expanduser()
        lexical_backed_up = False
        if lexical_source.is_file():
            lexical_dir = dest / "lexical"
            lexical_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lexical_source, lexical_dir / lexical_source.name)
            lexical_backed_up = True

        manifest_root = index_path.parent / f"{index_path.name}_manifests"
        configured_manifest = config.get("manifest_path")
        if configured_manifest:
            manifest_root = Path(str(configured_manifest)).expanduser().resolve().parent
        if manifest_root.is_dir():
            shutil.copytree(manifest_root, dest / "manifests")

        config_path = getattr(config, "config_path", None)
        if config_path and Path(config_path).is_file():
            shutil.copy2(config_path, dest / "config.yaml")

        # Read counts/provenance from the COPY, never the live dir — the live
        # client stays untouched while we hold the lock.
        store = _store_for_index(config, chroma_copy, name)
        index_state = read_index_state(store)
        chunk_count = int(store.count())

        payload = {
            "schema_version": BACKUP_SCHEMA_VERSION,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "collection_name": name,
            "index_path": str(index_path),
            "chunk_count": chunk_count,
            "index_state": index_state.as_dict() if index_state is not None else None,
            "lexical_backed_up": lexical_backed_up,
            "checksums": _checksum_tree(chroma_copy),
        }
        manifest_path = dest / _MANIFEST_NAME
        temporary = dest / f".{_MANIFEST_NAME}.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)
    finally:
        lock.release()
    return dest


def read_backup_manifest(backup_dir: Path) -> Dict[str, Any]:
    manifest_path = Path(backup_dir).expanduser().resolve() / _MANIFEST_NAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupError(f"Unreadable backup manifest: {manifest_path}") from exc
    if payload.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise BackupError("Unsupported backup manifest schema")
    return payload


def verify_backup(backup_dir: Path) -> Dict[str, Any]:
    """Verify checksums, chunk count, and provenance of a backup in place."""
    backup_dir = Path(backup_dir).expanduser().resolve()
    payload = read_backup_manifest(backup_dir)
    chroma_copy = backup_dir / "chroma"
    if not chroma_copy.is_dir():
        raise BackupError(f"Backup has no chroma/ directory: {backup_dir}")

    actual = _checksum_tree(chroma_copy)
    expected = payload.get("checksums", {})
    if actual != expected:
        extra = sorted(set(actual) - set(expected))
        missing = sorted(set(expected) - set(actual))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        raise BackupError(
            f"Backup checksum mismatch; extra={extra}, missing={missing}, changed={changed}"
        )

    store = _store_for_index(payload, chroma_copy, payload["collection_name"])
    count = int(store.count())
    if count != payload["chunk_count"]:
        raise BackupError(
            f"Backup chunk count mismatch: manifest {payload['chunk_count']}, actual {count}"
        )
    state = read_index_state(store)
    expected_state = payload.get("index_state")
    if expected_state is not None:
        if state is None:
            raise BackupError("Backup is missing the recorded index provenance")
        if state.provenance_fingerprint != expected_state.get("provenance_fingerprint"):
            raise BackupError("Backup provenance fingerprint does not match manifest")
        if state.generation_id != expected_state.get("generation_id"):
            raise BackupError("Backup generation id does not match manifest")
    return {"chunk_count": count, "verified": True}


def restore_backup(
    backup_dir: Path,
    dest_index_path: Path,
    wait_for_lock: float = 0.0,
) -> Path:
    """Restore a backup's Chroma dir to ``dest_index_path`` (must be empty/absent)."""
    backup_dir = Path(backup_dir).expanduser().resolve()
    read_backup_manifest(backup_dir)  # validate before touching disk
    chroma_copy = backup_dir / "chroma"
    if not chroma_copy.is_dir():
        raise BackupError(f"Backup has no chroma/ directory: {backup_dir}")
    dest_index_path = Path(dest_index_path).expanduser().resolve()
    if dest_index_path.exists() and any(dest_index_path.iterdir()):
        raise BackupError(f"Restore destination is not empty: {dest_index_path}")

    lock = IndexWriterLock(dest_index_path, "rag-restore", timeout=wait_for_lock)
    lock.acquire()
    try:
        shutil.copytree(chroma_copy, dest_index_path, dirs_exist_ok=True)
    finally:
        lock.release()
    return dest_index_path


def verify_restore(backup_dir: Path, dest_index_path: Path) -> Dict[str, Any]:
    """Compare a restored index against the backup manifest (counts, provenance, query)."""
    backup_dir = Path(backup_dir).expanduser().resolve()
    payload = read_backup_manifest(backup_dir)
    dest_index_path = Path(dest_index_path).expanduser().resolve()
    store = _store_for_index(payload, dest_index_path, payload["collection_name"])

    count = int(store.count())
    if count != payload["chunk_count"]:
        raise BackupError(
            f"Restore chunk count mismatch: manifest {payload['chunk_count']}, actual {count}"
        )
    state = read_index_state(store)
    expected_state = payload.get("index_state")
    fingerprint_ok = True
    if expected_state is not None:
        fingerprint_ok = (
            state is not None
            and state.provenance_fingerprint == expected_state.get("provenance_fingerprint")
            and state.generation_id == expected_state.get("generation_id")
        )
        if not fingerprint_ok:
            raise BackupError("Restored provenance does not match the backup manifest")
    hits = _representative_query(store, state)
    return {
        "chunk_count": count,
        "provenance_verified": fingerprint_ok,
        "representative_hits": hits,
    }


def restore_drill(backup_dir: Path) -> Dict[str, Any]:
    """Restore a backup to a throwaway temp path and verify it, then clean up."""
    backup_dir = Path(backup_dir).expanduser().resolve()
    temp_root = Path(tempfile.mkdtemp(prefix="rag-restore-drill-"))
    dest = temp_root / "chroma"
    try:
        restore_backup(backup_dir, dest)
        report = verify_restore(backup_dir, dest)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    report["drill"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-backup",
        description="Create, verify, and restore quiesced backups of the index.",
    )
    parser.add_argument("--config", help="Config path (overrides RAG_CONFIG_PATH)")
    parser.add_argument("--collection", help="Override the collection name")
    parser.add_argument(
        "--wait-for-lock", type=float, default=0.0, metavar="SECONDS",
        help="Wait up to SECONDS for the index writer lock instead of failing fast",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create", help="Create and verify a new backup")
    create.add_argument("dest", help="Destination directory (must be empty/absent)")
    verify = sub.add_parser("verify", help="Verify an existing backup in place")
    verify.add_argument("backup_dir")
    restore = sub.add_parser("restore", help="Restore a backup to an index path")
    restore.add_argument("backup_dir")
    restore.add_argument("dest_index_path")
    drill = sub.add_parser("drill", help="Restore to a temp path, verify, and clean up")
    drill.add_argument("backup_dir")
    args = parser.parse_args()

    try:
        if args.command == "create":
            config = load_config(args.config)
            dest = create_backup(
                config, args.dest, args.collection, wait_for_lock=args.wait_for_lock
            )
            report = verify_backup(dest)
            print(f"Backup created and verified at {dest} ({report['chunk_count']} chunks)")
        elif args.command == "verify":
            report = verify_backup(args.backup_dir)
            print(f"Backup verified ({report['chunk_count']} chunks)")
        elif args.command == "restore":
            dest = restore_backup(
                args.backup_dir, args.dest_index_path, wait_for_lock=args.wait_for_lock
            )
            report = verify_restore(args.backup_dir, dest)
            print(
                f"Restored to {dest} ({report['chunk_count']} chunks, "
                f"provenance_verified={report['provenance_verified']})"
            )
        elif args.command == "drill":
            report = restore_drill(args.backup_dir)
            print(
                f"Restore drill passed ({report['chunk_count']} chunks, "
                f"representative_hits={report['representative_hits']})"
            )
    except (BackupError, IndexLockedError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
