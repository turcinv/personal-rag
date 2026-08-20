from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ARTIFACT_SCHEMA_VERSION = 1
_REQUIRED_PATHS = {
    "text/books/manifest.json",
    "text/resources/manifest.json",
    "catalog/resource_inventory_enriched.jsonl",
    "indexed/index_documents.jsonl",
    "indexed/vault_documents.jsonl",
    "indexed/build_report.json",
    "resources.db",
}
_ALIASES = {
    "indexed": Path("current/indexed"),
    "resources.db": Path("current/resources.db"),
    "text_output_books": Path("current/text/books"),
    "text_output_resources": Path("current/text/resources"),
}


class ArtifactError(RuntimeError):
    """Raised when an artifact generation is incomplete or unsafe to publish."""


@dataclass(frozen=True)
class ArtifactGeneration:
    root: Path
    generation_id: str
    staging: Path
    parent_generation_id: Optional[str]

    @property
    def final(self) -> Path:
        return self.root / "generations" / self.generation_id


def _root(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _validated_current(root: Path) -> Optional[Path]:
    root = _root(root)
    current = root / "current"
    if not _exists(current):
        return None
    if not current.is_symlink():
        raise ArtifactError(f"Artifact pointer is not a symlink: {current}")
    target = (current.parent / os.readlink(current)).resolve()
    generations = (root / "generations").resolve()
    if target.parent != generations:
        raise ArtifactError(f"Artifact pointer escapes the generations directory: {current}")
    validate_generation(target, expected_generation_id=target.name)
    return target


def active_generation_id(root: Path) -> Optional[str]:
    target = _validated_current(root)
    if target is None:
        return None
    payload = validate_generation(target)
    return str(payload["generation_id"])


def preflight_publication(root: Path) -> None:
    """Reject unsafe legacy layouts before building or activating a generation."""
    root = _root(root)
    _validated_current(root)
    for name, target in _ALIASES.items():
        destination = root / name
        if not _exists(destination):
            continue
        if not destination.is_symlink():
            raise ArtifactError(
                f"Cannot create compatibility alias {destination}: a legacy path "
                "already exists; move it aside after backup, then rerun"
            )
        if Path(os.readlink(destination)) != target:
            raise ArtifactError(
                f"Compatibility alias {destination} has an unexpected target; "
                "move it aside after backup, then rerun"
            )


def begin_generation(root: Path) -> ArtifactGeneration:
    root = _root(root)
    preflight_publication(root)
    generations = root / "generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    staging = generations / f".staging-{generation_id}"
    staging.mkdir()
    return ArtifactGeneration(root, generation_id, staging, active_generation_id(root))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _role(relative: str) -> str:
    if relative.startswith("indexed/") and relative.endswith(".json"):
        if relative != "indexed/build_report.json":
            return "indexed_document"
    if relative.startswith("text/") and relative.endswith(".json"):
        return "extraction"
    if relative.endswith(".jsonl"):
        return "jsonl"
    if relative.endswith(".db"):
        return "sqlite"
    return "support"


def _sqlite_count(path: Path) -> int:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise ArtifactError(f"SQLite quick_check failed for {path}: {check}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        required = {"documents", "documents_fts"}
        if not required <= tables:
            raise ArtifactError(
                f"SQLite artifact {path} missing tables: {sorted(required - tables)}"
            )
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        fts = connection.execute("SELECT COUNT(*) FROM documents_fts").fetchone()[0]
        if documents != fts:
            raise ArtifactError(
                f"SQLite document/FTS count mismatch: {documents} != {fts}"
            )
        return int(documents)
    finally:
        connection.close()


def _record_count(path: Path) -> int:
    try:
        if path.suffix == ".jsonl":
            count = 0
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        json.loads(line)
                        count += 1
            return count
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "manifest.json" and isinstance(payload, list):
                failures = [
                    item
                    for item in payload
                    if isinstance(item, dict) and item.get("status") == "error"
                ]
                if failures:
                    raise ArtifactError(
                        f"Extraction manifest contains {len(failures)} failed file(s): {path}"
                    )
            return len(payload) if isinstance(payload, list) else 1
        if path.suffix == ".db":
            return _sqlite_count(path)
    except (OSError, ValueError, TypeError, sqlite3.DatabaseError) as exc:
        if isinstance(exc, ArtifactError):
            raise
        raise ArtifactError(f"Invalid artifact {path}: {exc}") from exc
    return 0


def _artifact_files(path: Path) -> List[Path]:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ArtifactError(f"Artifact generations cannot contain symlinks: {item}")
        if not item.is_file():
            continue
        if item.name.endswith(("-wal", "-shm")):
            raise ArtifactError(f"SQLite sidecar must not be published: {item}")
        files.append(item)
    return files


def build_manifest(generation: ArtifactGeneration, profile: str = "") -> dict:
    artifacts: List[dict] = []
    for path in _artifact_files(generation.staging):
        relative = path.relative_to(generation.staging).as_posix()
        if relative in {"manifest.json", ".manifest.tmp"}:
            continue
        artifacts.append(
            {
                "path": relative,
                "role": _role(relative),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "records": _record_count(path),
            }
        )
    paths = {artifact["path"] for artifact in artifacts}
    missing = sorted(_REQUIRED_PATHS - paths)
    if missing:
        raise ArtifactError(f"Generation is incomplete; missing: {', '.join(missing)}")

    parent_paths = set()
    if generation.parent_generation_id:
        parent = generation.root / "generations" / generation.parent_generation_id
        parent_payload = validate_generation(parent)
        parent_paths = {item["path"] for item in parent_payload["artifacts"]}

    counts: Dict[str, int] = {}
    for artifact in artifacts:
        role = artifact["role"]
        counts[role] = counts.get(role, 0) + int(artifact["records"])
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "generation_id": generation.generation_id,
        "parent_generation_id": generation.parent_generation_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": profile,
        "complete": True,
        "counts": counts,
        "removed_since_parent": sorted(parent_paths - paths),
        "artifacts": artifacts,
    }


def write_manifest(generation: ArtifactGeneration, profile: str = "") -> Path:
    payload = build_manifest(generation, profile=profile)
    path = generation.staging / "manifest.json"
    temporary = generation.staging / ".manifest.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    validate_generation(generation.staging, expected_generation_id=generation.generation_id)
    return path


def validate_generation(
    path: Path, expected_generation_id: Optional[str] = None
) -> dict:
    path = Path(path).expanduser().resolve()
    manifest_path = path / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"Unreadable artifact manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("Artifact manifest must be a JSON object")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("Unsupported artifact manifest schema")
    if payload.get("complete") is not True:
        raise ArtifactError("Artifact generation is not complete")
    generation_id = payload.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ArtifactError("Artifact manifest has no generation ID")
    if expected_generation_id is not None and generation_id != expected_generation_id:
        raise ArtifactError("Artifact manifest generation ID does not match its directory")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ArtifactError("Artifact manifest artifacts must be a list")

    listed = set()
    recomputed_counts: Dict[str, int] = {}
    for artifact in raw_artifacts:
        if not isinstance(artifact, dict):
            raise ArtifactError("Artifact manifest entries must be objects")
        relative = artifact.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ArtifactError(f"Unsafe manifest artifact path: {relative!r}")
        if relative in listed:
            raise ArtifactError(f"Duplicate manifest artifact path: {relative}")
        listed.add(relative)
        artifact_path = path / relative
        resolved = artifact_path.resolve()
        try:
            resolved.relative_to(path)
        except ValueError as exc:
            raise ArtifactError(f"Artifact escapes generation: {relative}") from exc
        if not artifact_path.is_file() or artifact_path.is_symlink():
            raise ArtifactError(f"Missing or unsafe artifact: {relative}")
        try:
            expected_bytes = int(artifact["bytes"])
            expected_records = int(artifact.get("records", 0))
            expected_sha = str(artifact["sha256"])
            role = str(artifact["role"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"Malformed manifest entry: {relative}") from exc
        if artifact_path.stat().st_size != expected_bytes:
            raise ArtifactError(f"Artifact size mismatch: {relative}")
        if sha256_file(artifact_path) != expected_sha:
            raise ArtifactError(f"Artifact checksum mismatch: {relative}")
        if _record_count(artifact_path) != expected_records:
            raise ArtifactError(f"Artifact record-count mismatch: {relative}")
        recomputed_counts[role] = recomputed_counts.get(role, 0) + expected_records

    actual = {
        relative
        for relative in (
            item.relative_to(path).as_posix() for item in _artifact_files(path)
        )
        if relative != "manifest.json"
    }
    if actual != listed:
        extra = sorted(actual - listed)
        missing = sorted(listed - actual)
        raise ArtifactError(f"Manifest file-set mismatch; extra={extra}, missing={missing}")
    if not _REQUIRED_PATHS <= listed:
        raise ArtifactError("Manifest does not list every required artifact")
    if payload.get("counts") != recomputed_counts:
        raise ArtifactError("Artifact manifest aggregate counts do not match its entries")
    return payload


def _install_aliases(root: Path) -> None:
    for name, target in _ALIASES.items():
        destination = root / name
        if destination.is_symlink() and Path(os.readlink(destination)) == target:
            continue
        temporary = root / f".{name}.next"
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, destination)


def publish_generation(generation: ArtifactGeneration, profile: str = "") -> Path:
    preflight_publication(generation.root)
    if generation.final.exists():
        raise ArtifactError(f"Artifact generation already exists: {generation.final}")
    write_manifest(generation, profile=profile)
    validate_generation(
        generation.staging, expected_generation_id=generation.generation_id
    )

    os.replace(generation.staging, generation.final)
    try:
        _install_aliases(generation.root)
        current_next = generation.root / ".current.next"
        current_next.unlink(missing_ok=True)
        current_next.symlink_to(Path("generations") / generation.generation_id)
        os.replace(current_next, generation.root / "current")
    except Exception:
        (generation.root / ".current.next").unlink(missing_ok=True)
        raise
    return generation.final


def resolve_active(root: Path) -> Path:
    target = _validated_current(root)
    if target is None:
        raise ArtifactError(f"No active artifact generation under {root}")
    return target
