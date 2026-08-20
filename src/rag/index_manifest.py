"""Source-scoped prune planning and atomic run manifests."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

from .indexing import SourceRunResult


@dataclass(frozen=True)
class SourcePrunePlan:
    source_id: str
    state: str
    existing_chunks: int
    seen_chunks: int
    new_chunks: int
    updated_chunks: int
    proposed_deletion_count: int
    preserved_chunks: int
    prune_eligible: bool
    blocked: bool
    reason: str = ""


@dataclass(frozen=True)
class PrunePlan:
    run_id: str
    created_at: str
    dry_run: bool
    sources: Tuple[SourcePrunePlan, ...]
    legacy_chunks_preserved: int

    @property
    def proposed_deletion_count(self) -> int:
        return sum(source.proposed_deletion_count for source in self.sources)

    @property
    def approved_deletion_count(self) -> int:
        return sum(
            source.proposed_deletion_count
            for source in self.sources
            if not source.blocked
        )

    def as_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "dry_run": self.dry_run,
            "legacy_chunks_preserved": self.legacy_chunks_preserved,
            "proposed_deletion_count": self.proposed_deletion_count,
            "applied_deletion_count": (
                0 if self.dry_run else self.approved_deletion_count
            ),
            "sources": [asdict(source) for source in self.sources],
        }


def build_prune_plan(
    reconciliation,
    results: Sequence[SourceRunResult],
    *,
    max_fraction: float,
    max_chunks: int,
    allow_large_prune: bool = False,
    allow_empty_source_prune: bool = False,
    dry_run: bool = False,
) -> PrunePlan:
    """Plan source deletions from disk-backed reconciliation aggregates."""
    source_ids = [result.source_id for result in results]
    if len(source_ids) != len(set(source_ids)):
        duplicates = sorted(
            source_id for source_id in set(source_ids) if source_ids.count(source_id) > 1
        )
        raise ValueError(f"Duplicate source result id(s): {', '.join(duplicates)}")

    owned = dict(reconciliation.owned_sources())
    result_by_id = {result.source_id: result for result in results}
    plans: List[SourcePrunePlan] = []

    for result in results:
        existing, seen, stale = reconciliation.source_counts(result.source_id)
        eligible = result.prune_eligible or (
            allow_empty_source_prune and result.state == "empty"
        )
        reason = result.detail
        blocked = not eligible
        if blocked and not reason:
            reason = f"source state {result.state!r} is not prune-eligible"

        fraction = (stale / existing) if existing else 0.0
        over_threshold = stale > max_chunks or fraction > max_fraction
        if eligible and stale and over_threshold and not allow_large_prune:
            blocked = True
            reason = (
                f"deletion threshold exceeded: {stale} chunks, "
                f"{fraction:.1%} of source; use --allow-large-prune after review"
            )

        plans.append(
            SourcePrunePlan(
                source_id=result.source_id,
                state=result.state,
                existing_chunks=existing,
                seen_chunks=seen,
                new_chunks=result.new_chunks,
                updated_chunks=result.updated_chunks,
                proposed_deletion_count=stale,
                preserved_chunks=stale if blocked else 0,
                prune_eligible=eligible,
                blocked=blocked,
                reason=reason,
            )
        )

    for source_id in sorted(set(owned) - set(result_by_id)):
        existing, seen, stale = reconciliation.source_counts(source_id)
        plans.append(
            SourcePrunePlan(
                source_id=source_id,
                state="not-configured",
                existing_chunks=existing,
                seen_chunks=seen,
                new_chunks=0,
                updated_chunks=0,
                proposed_deletion_count=stale,
                preserved_chunks=stale,
                prune_eligible=False,
                blocked=True,
                reason=(
                    "source ownership exists in the index but source was not "
                    "configured this run"
                ),
            )
        )

    return PrunePlan(
        run_id=f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        dry_run=dry_run,
        sources=tuple(plans),
        legacy_chunks_preserved=reconciliation.legacy_count(),
    )


def apply_prune_plan(store, plan: PrunePlan, reconciliation, batch_size: int = 500) -> int:
    """Stream approved source deletions in bounded batches."""
    deleted = 0
    for source in plan.sources:
        if source.blocked or not source.proposed_deletion_count:
            continue
        for ids in reconciliation.iter_stale_batches(source.source_id, batch_size):
            store.delete(ids)
            deleted += len(ids)
    return deleted


def manifest_directory(config: Mapping[str, object], collection_name: str) -> Path:
    configured = config.get("manifest_path")
    if configured:
        root = Path(str(configured))
    else:
        index_path = Path(str(config.get("index_path", "./chroma_db")))
        root = index_path.parent / f"{index_path.name}_manifests"
    return root / collection_name


def write_run_manifest(config, collection_name: str, plan: PrunePlan) -> Path:
    """Persist one completed run manifest with an atomic file replacement."""
    directory = manifest_directory(config, collection_name)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{plan.run_id}.json"
    temporary = directory / f".{plan.run_id}.tmp"
    temporary.write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination
