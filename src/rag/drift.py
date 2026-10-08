"""Report-only drift check between the vault source and the retrieval index."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

from .extractors import iter_sources
from .indexing import _owned_metadata
from .reconciliation import ReconciliationCatalog
from .store import get_store
from .utils import load_config

DEFAULT_REMOTE = "fedora"


@dataclass
class GitStatus:
    """Result of the source-checkout-vs-upstream sub-check (never 'clean' on error)."""

    state: str  # "ok" | "unknown"
    is_git: bool = False
    remote: str = DEFAULT_REMOTE
    ahead: Optional[int] = None
    behind: Optional[int] = None
    dirty: Optional[int] = None
    detail: str = ""

    @property
    def drift(self) -> bool:
        """A git sub-check contributes drift only when it positively observed it."""
        if self.state != "ok":
            return False
        return bool((self.behind or 0) > 0 or (self.dirty or 0) > 0)

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "is_git": self.is_git,
            "remote": self.remote,
            "ahead": self.ahead,
            "behind": self.behind,
            "dirty": self.dirty,
            "detail": self.detail,
        }


@dataclass
class SourceDrift:
    """Per-source reconciliation outcome (source files vs indexed chunks)."""

    source_id: str
    state: str  # source enumeration state ("available", "missing", ...)
    missing_files: List[str] = field(default_factory=list)
    changed_files: List[str] = field(default_factory=list)
    stale_chunks: int = 0
    detail: str = ""

    @property
    def drift(self) -> bool:
        return bool(self.missing_files or self.changed_files or self.stale_chunks)

    def as_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "state": self.state,
            "missing_files": self.missing_files,
            "changed_files": self.changed_files,
            "stale_chunks": self.stale_chunks,
            "detail": self.detail,
        }


def _run_git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git command read-only; return stdout stripped, or None on any failure."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def check_git_upstream(vault_path: Path, remote: str = DEFAULT_REMOTE) -> GitStatus:
    """Compare the vault checkout against ``<remote>/main`` (read-only).

    Any git failure is reported as "unknown", never silently as "no drift".
    """
    inside = _run_git(["rev-parse", "--is-inside-work-tree"], vault_path)
    if inside != "true":
        return GitStatus(
            state="unknown",
            is_git=False,
            remote=remote,
            detail="vault_path is not a git work tree",
        )

    # Fetch is read-only; a failure (offline, no such remote) leaves us unknown.
    fetched = _run_git(["fetch", "--quiet", remote], vault_path)
    porcelain = _run_git(["status", "--porcelain"], vault_path)
    dirty = None if porcelain is None else len([ln for ln in porcelain.splitlines() if ln.strip()])

    counts = _run_git(
        ["rev-list", "--left-right", "--count", f"HEAD...{remote}/main"],
        vault_path,
    )
    if counts is None:
        return GitStatus(
            state="unknown",
            is_git=True,
            remote=remote,
            dirty=dirty,
            detail=(
                f"upstream not configured or unreachable ({remote}/main); "
                "fetch " + ("ok" if fetched is not None else "failed")
            ),
        )
    try:
        ahead_s, behind_s = counts.split()
        ahead, behind = int(ahead_s), int(behind_s)
    except (ValueError, IndexError):
        return GitStatus(
            state="unknown",
            is_git=True,
            remote=remote,
            dirty=dirty,
            detail=f"could not parse ahead/behind from {counts!r}",
        )
    return GitStatus(
        state="ok",
        is_git=True,
        remote=remote,
        ahead=ahead,
        behind=behind,
        dirty=dirty,
    )


def check_source_drift(config: dict, source_id: str = "markdown:vault") -> SourceDrift:
    """Reconcile one source's files against the index, read-only and embedding-free."""
    vault_path = Path(config["vault_path"])
    max_chars = int(config.get("chunk_max_chars", 1200))
    overlap = int(config.get("chunk_overlap_chars", 150))
    collection_name = config.get("collection_name", "obsidian_markdown")
    index_path = Path(config.get("index_path", "./chroma_db"))

    sources = {s.source_id: s for s in iter_sources(config, vault_path, max_chars, overlap)}
    source = sources.get(source_id)
    if source is None:
        return SourceDrift(
            source_id=source_id,
            state="unknown",
            detail=f"no such source; available: {', '.join(sorted(sources)) or 'none'}",
        )
    if source.state != "available" or source.extract is None:
        return SourceDrift(
            source_id=source_id,
            state=source.state,
            detail=source.detail or "source not available for enumeration",
        )

    missing_files: List[str] = []
    changed_files: List[str] = []
    store = get_store(config, collection_name)
    with ReconciliationCatalog.from_store(store, index_path) as catalog:
        # file_ids this source already owns in the index BEFORE this pass. A file
        # whose chunks all re-hash to new IDs (a fully-rewritten note) is
        # indistinguishable from a brand-new file by chunk ID alone — the only
        # signal that it CHANGED rather than APPEARED is that the index already
        # held chunks for its file_id.
        indexed_file_ids = {
            row[0]
            for row in catalog.connection.execute(
                "SELECT DISTINCT file_id FROM chunks "
                "WHERE was_existing = 1 AND original_source_id = ? AND file_id IS NOT NULL",
                (source.source_id,),
            )
        }
        for path in source.files:
            ids, docs, metadatas, error = source.extract(path)
            file_id = source.file_key(path)
            if error or not docs:
                # Mirror the indexer: a failed/empty file preserves prior chunks,
                # so it is not drift. Keep those chunks out of the stale bucket.
                catalog.preserve_file(source.source_id, file_id)
                continue
            owned = _owned_metadata(metadatas, source, file_id)
            new_indices, updated_indices = catalog.classify_and_mark(ids, owned)
            if not (new_indices or updated_indices):
                continue  # every chunk matched the index exactly — in sync
            if file_id in indexed_file_ids:
                # The file was already indexed and now produces new/updated chunks
                # (edited body ⇒ new IDs + old ones stale, or metadata-only change).
                changed_files.append(file_id)
            else:
                # No prior chunks under this file_id ⇒ a file absent from the index.
                missing_files.append(file_id)
        _, _, stale = catalog.source_counts(source.source_id)

    return SourceDrift(
        source_id=source_id,
        state="available",
        missing_files=sorted(missing_files),
        changed_files=sorted(changed_files),
        stale_chunks=int(stale),
    )


def _format_report(git: GitStatus, sources: List[SourceDrift], generated_at: str) -> str:
    lines = [f"Drift report — {generated_at}", ""]
    lines.append("Source checkout vs upstream:")
    if git.state == "unknown":
        lines.append(f"  unknown — {git.detail}")
    elif not git.is_git and git.detail.startswith("skipped"):
        lines.append(f"  {git.detail}")
    else:
        flags = []
        if (git.behind or 0) > 0:
            flags.append(f"BEHIND {git.behind}")
        if (git.ahead or 0) > 0:
            flags.append(f"ahead {git.ahead}")
        if (git.dirty or 0) > 0:
            flags.append(f"dirty {git.dirty}")
        verdict = ", ".join(flags) if flags else "in sync"
        lines.append(f"  {git.remote}/main: {verdict}")
    lines.append("")
    for sd in sources:
        lines.append(f"Source {sd.source_id} ({sd.state}):")
        if sd.state != "available":
            lines.append(f"  skipped — {sd.detail}")
            continue
        lines.append(f"  missing from index: {len(sd.missing_files)}")
        for name in sd.missing_files[:20]:
            lines.append(f"    + {name}")
        if len(sd.missing_files) > 20:
            lines.append(f"    … and {len(sd.missing_files) - 20} more")
        lines.append(f"  content changed:    {len(sd.changed_files)}")
        for name in sd.changed_files[:20]:
            lines.append(f"    ~ {name}")
        if len(sd.changed_files) > 20:
            lines.append(f"    … and {len(sd.changed_files) - 20} more")
        lines.append(f"  stale in index:     {sd.stale_chunks} chunk(s)")
    return "\n".join(lines)


def run_drift(
    config: dict,
    source_ids: List[str],
    remote: str = DEFAULT_REMOTE,
    skip_git: bool = False,
):
    """Run every drift sub-check and return ``(git_status, [source_drift, ...])``.

    ``skip_git=True`` reports the git sub-check as "skipped" (state "ok", not
    "unknown") so it neither contributes drift nor forces could-not-verify — for
    callers that run the git comparison elsewhere, e.g. the Jetson container
    where git/the bare repo/a writable vault are all unavailable.
    """
    if skip_git:
        git = GitStatus(
            state="ok",
            is_git=False,
            remote=remote,
            detail="skipped (--skip-git); git checked on the host",
        )
    else:
        git = check_git_upstream(Path(config["vault_path"]), remote=remote)
    sources = [check_source_drift(config, source_id=sid) for sid in source_ids]
    return git, sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report drift between the vault source and the retrieval index "
        "(read-only; embeds nothing, never mutates the index).",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="ID",
        help="Source id to check (repeatable). Default: markdown:vault.",
    )
    parser.add_argument(
        "--remote",
        default=DEFAULT_REMOTE,
        help=f"Git remote to compare the vault checkout against (default: {DEFAULT_REMOTE}).",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip the git-upstream sub-check (reported 'skipped', never drift/exit 2). "
        "For environments without git/the bare repo/a writable vault (e.g. the Jetson "
        "container); run the git comparison on the host instead.",
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of a table.",
    )
    args = parser.parse_args()

    config = load_config()
    source_ids = args.source or ["markdown:vault"]
    git, sources = run_drift(config, source_ids, remote=args.remote, skip_git=args.skip_git)

    drift = git.drift or any(sd.drift for sd in sources)

    # A sub-check that could not actually run (git "unknown", or a source that
    # was not "available" for enumeration — e.g. the Jetson vault-mount failure)
    # must NOT read as a green "no drift". Collect those reasons; drift (exit 1)
    # still wins over could-not-verify (exit 2), because an observed drift is a
    # strictly stronger signal than an unverifiable sub-check.
    unverifiable = []
    if git.state != "ok":
        unverifiable.append(f"git check: {git.detail or 'unknown'}")
    for sd in sources:
        if sd.state != "available":
            unverifiable.append(f"source {sd.source_id}: {sd.detail or sd.state}")

    if drift:
        exit_code = 1
    elif unverifiable:
        exit_code = 2
    else:
        exit_code = 0

    generated_at = datetime.now(UTC).isoformat()
    if args.output_json:
        print(
            json.dumps(
                {
                    "generated_at": generated_at,
                    "drift": drift,
                    "could_not_verify": bool(unverifiable) and not drift,
                    "unverifiable": unverifiable,
                    "git": git.as_dict(),
                    "sources": [sd.as_dict() for sd in sources],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(_format_report(git, sources, generated_at))
        print()
        if drift:
            print("DRIFT DETECTED")
        elif unverifiable:
            print("COULD NOT VERIFY — " + "; ".join(unverifiable))
        else:
            print("no drift")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
