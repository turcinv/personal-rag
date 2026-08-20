"""Bounded, source-scoped incremental indexing engine."""

import gc
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

logger = logging.getLogger("rag")
log = logger.info


@dataclass(frozen=True)
class SourceRunResult:
    source_id: str
    state: str
    file_count: int
    succeeded_files: int
    failed_files: int
    empty_files: int
    total_chunks: int
    new_chunks: int
    updated_chunks: int
    detail: str = ""

    @property
    def prune_eligible(self) -> bool:
        return (
            self.state == "completed"
            and self.file_count > 0
            and self.failed_files == 0
            and self.empty_files == 0
        )

    def __iter__(self):
        """Preserve historical ``total, new, updated = run_source(...)`` use."""
        yield self.total_chunks
        yield self.new_chunks
        yield self.updated_chunks


def embed_and_upsert(model, device, docs, ids, metas, embed_batch_size, store):
    """Embed in small batches and upsert immediately."""
    count = len(docs)
    batch_count = (count + embed_batch_size - 1) // embed_batch_size
    for batch_index, start in enumerate(range(0, count, embed_batch_size), 1):
        batch_docs = docs[start:start + embed_batch_size]
        batch_ids = ids[start:start + embed_batch_size]
        batch_metas = metas[start:start + embed_batch_size]
        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            batch_size=embed_batch_size,
        )
        store.upsert(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            docs=batch_docs,
            metas=batch_metas,
        )
        del embeddings
        if device == "cuda":
            torch.cuda.empty_cache()
        if batch_count > 1:
            log(
                f"      batch {batch_index}/{batch_count}  "
                f"({min(start + embed_batch_size, count)}/{count} chunks)"
            )


def index_file_chunks(
    ids,
    docs,
    metas,
    reconciliation,
    model,
    device,
    embed_batch,
    store,
    dry_run=False,
):
    """Classify one file's chunks and optionally apply inserts/metadata updates."""
    new_indices, updated_indices = reconciliation.classify_and_mark(ids, metas)

    if new_indices and not dry_run:
        embed_and_upsert(
            model,
            device,
            [docs[index] for index in new_indices],
            [ids[index] for index in new_indices],
            [metas[index] for index in new_indices],
            embed_batch,
            store,
        )
    if updated_indices and not dry_run:
        store.update_metadata(
            ids=[ids[index] for index in updated_indices],
            metas=[metas[index] for index in updated_indices],
        )
    return len(new_indices), len(updated_indices), len(ids)


def preserve_existing(source_id, file_id, reconciliation):
    """Preserve prior chunks for one failed or unexpectedly empty file."""
    return reconciliation.preserve_file(source_id, file_id)


def _owned_metadata(metadatas, source, file_id):
    return [
        {
            **metadata,
            "source_id": source.source_id,
            "source_kind": source.kind,
            "file_id": file_id,
        }
        for metadata in metadatas
    ]


def _index_status(new, updated, total):
    if new or updated:
        return f"{new} new, {updated} meta / {total} chunks"
    return f"unchanged, {total} chunks"


def _bounded_extractions(source):
    """Yield extraction results in source order with at most ``workers`` futures."""
    window = max(1, int(source.workers))
    paths = iter(enumerate(source.files, 1))
    pending = deque()
    maximum_depth = 0

    with ThreadPoolExecutor(max_workers=window) as pool:
        for _ in range(window):
            try:
                index, path = next(paths)
            except StopIteration:
                break
            pending.append((index, path, pool.submit(source.extract, path)))
        maximum_depth = max(maximum_depth, len(pending))
        log(f"  extraction queue: workers={window}, depth={len(pending)}")

        while pending:
            index, path, future = pending.popleft()
            result = future.result()
            del future
            yield index, path, result
            del result
            try:
                next_index, next_path = next(paths)
            except StopIteration:
                continue
            pending.append(
                (next_index, next_path, pool.submit(source.extract, next_path))
            )
            maximum_depth = max(maximum_depth, len(pending))

    log(f"  extraction queue max depth: {maximum_depth}")


def run_source(
    source,
    reconciliation,
    model,
    device,
    embed_batch,
    store,
    dry_run=False,
):
    """Run one source and return its explicit reconciliation outcome."""
    log(f"\n{source.label}")
    if source.state != "available":
        return SourceRunResult(
            source.source_id,
            source.state,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            source.detail,
        )
    if not source.files:
        if source.excluded_files:
            detail = f"{source.excluded_files} file(s) intentionally delegated to JSON"
            return SourceRunResult(
                source.source_id,
                "completed",
                source.excluded_files,
                source.excluded_files,
                0,
                0,
                0,
                0,
                0,
                detail,
            )
        return SourceRunResult(
            source.source_id,
            "empty",
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            "source enumerated successfully but contained no files",
        )

    work_count = len(source.files)
    file_count = work_count + source.excluded_files
    total = new = updated = failed = empty = 0
    succeeded = source.excluded_files
    for index, path, extraction in _bounded_extractions(source):
        ids, docs, metadatas, error = extraction
        file_id = source.file_key(path)
        if error:
            failed += 1
            log(
                f"  [{index}/{work_count}] SKIP {path.name}: "
                f"{error.split(':', 1)[-1].strip()}"
            )
            preserve_existing(source.source_id, file_id, reconciliation)
            del ids, docs, metadatas, extraction
            continue
        if not docs:
            empty += 1
            log(
                f"  [{index}/{work_count}] PRESERVE {path.name}: "
                "extraction returned no chunks"
            )
            preserve_existing(source.source_id, file_id, reconciliation)
            del ids, docs, metadatas, extraction
            continue

        metadatas = _owned_metadata(metadatas, source, file_id)
        file_new, file_updated, file_total = index_file_chunks(
            ids,
            docs,
            metadatas,
            reconciliation,
            model,
            device,
            embed_batch,
            store,
            dry_run=dry_run,
        )
        succeeded += 1
        total += file_total
        new += file_new
        updated += file_updated
        log(
            f"  [{index}/{work_count}] {path.name}  "
            f"({_index_status(file_new, file_updated, file_total)})"
        )
        del ids, docs, metadatas, extraction
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    state = "completed" if failed == 0 and empty == 0 else "degraded"
    detail = ""
    if state == "degraded":
        detail = f"{failed} failed file(s), {empty} unexpectedly empty file(s)"
    log(
        f"  {state}: {total} chunks ({new} embedded, {updated} metadata-updated; "
        f"{failed} failed, {empty} empty)"
    )
    return SourceRunResult(
        source.source_id,
        state,
        file_count,
        succeeded,
        failed,
        empty,
        total,
        new,
        updated,
        detail,
    )
