"""Source-scoped incremental indexer for Markdown, PDF, and JSON."""

import argparse
import contextlib
import logging
from pathlib import Path

from .utils import load_config, setup_logging

import torch
from sentence_transformers import SentenceTransformer

from .extractors import iter_sources
from .index_manifest import apply_prune_plan, build_prune_plan, write_run_manifest
from .indexing import run_source
from .provenance import (
    begin_generation,
    ensure_compatible,
    expected_provenance,
    publish_generation,
    read_index_state,
)
from .reconciliation import ReconciliationCatalog
from .locking import IndexLockedError, IndexWriterLock
from .store import get_store


logger = logging.getLogger("rag")
log = logger.info


def _log_prune_plan(plan) -> None:
    log("\nSource-scoped reconciliation plan:")
    for source in plan.sources:
        action = "PRESERVE" if source.blocked else "PRUNE"
        log(
            f"  {source.source_id}: state={source.state} existing={source.existing_chunks} "
            f"seen={source.seen_chunks} new={source.new_chunks} "
            f"updated={source.updated_chunks} proposed_delete="
            f"{source.proposed_deletion_count} action={action}"
        )
        if source.reason:
            log(f"    reason: {source.reason}")
    if plan.legacy_chunks_preserved:
        log(
            f"  legacy/unowned chunks preserved: {plan.legacy_chunks_preserved} "
            "(they become source-owned when successfully seen again)"
        )


def _validate_unique_sources(sources) -> None:
    locations: dict = {}
    for index, source in enumerate(sources):
        previous = locations.get(source.source_id)
        if previous is not None:
            raise ValueError(
                f"Duplicate source id {source.source_id!r} at positions "
                f"{previous} and {index}; source IDs must be globally unique"
            )
        locations[source.source_id] = index


def main():
    parser = argparse.ArgumentParser(description="Index vault + PDFs + JSON into the retrieval store.")
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Override the collection name from config.yaml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and print insert/update/delete plans without mutating the index",
    )
    parser.add_argument(
        "--allow-large-prune",
        action="store_true",
        help="Allow reviewed source deletions above configured count/fraction thresholds",
    )
    parser.add_argument(
        "--allow-empty-source-prune",
        action="store_true",
        help="Treat a successfully enumerated zero-file source as an intentional deletion",
    )
    parser.add_argument(
        "--adopt-index-provenance",
        action="store_true",
        help=(
            "Attach the active profile provenance to a legacy non-empty index; "
            "use only after independently verifying how its vectors were built"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        help="Override the directory used for index-run manifests",
    )
    parser.add_argument(
        "--wait-for-lock",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Wait up to SECONDS for the index writer lock instead of failing fast",
    )
    args = parser.parse_args()
    if args.dry_run and args.adopt_index_provenance:
        parser.error("--adopt-index-provenance cannot be used with --dry-run")

    config = load_config()
    if args.manifest_path:
        config["manifest_path"] = str(Path(args.manifest_path).expanduser().resolve())
    setup_logging(config)

    vault_path = Path(config["vault_path"])
    collection_name = args.collection or config.get("collection_name", "obsidian_markdown")
    max_chars = int(config.get("chunk_max_chars", 1200))
    overlap = int(config.get("chunk_overlap_chars", 150))
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch = int(config.get("embedding_batch_size", 16))
    md_workers = int(config.get("markdown_workers", 1))
    pdf_workers = int(config.get("pdf_workers", 1))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Vault: {vault_path}")
    log(f"Collection: {collection_name}  |  metric: cosine")
    log(
        f"Embedding model: {model_name}  |  device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else "")
    )
    log(f"Chunk max chars: {max_chars}  |  overlap: {overlap}")
    log(
        f"Embed batch size: {embed_batch}  |  md_workers: {md_workers}  "
        f"|  pdf_workers: {pdf_workers}"
    )
    sources = list(iter_sources(config, vault_path, max_chars, overlap))
    _validate_unique_sources(sources)
    total_files = sum(len(source.files) + source.excluded_files for source in sources)
    for source in sources:
        log(
            f"Source {source.source_id}: state={source.state}, "
            f"files={len(source.files)}, delegated={source.excluded_files}, "
            f"root={source.root}"
        )
    log(f"Total source files found: {total_files}")

    index_path = Path(config.get("index_path", "./chroma_db"))
    with contextlib.ExitStack() as stack:
        # A full run mutates the store (generation bump, upserts, deletes) and
        # writes a run manifest, so it must hold the cross-process writer lock.
        # A dry run is read-only planning and takes no lock.
        if not args.dry_run:
            try:
                writer_lock = stack.enter_context(
                    IndexWriterLock(
                        index_path, "rag-index", timeout=args.wait_for_lock
                    )
                )
            except IndexLockedError as exc:
                raise SystemExit(str(exc))
            if writer_lock.reclaimed_from is not None:
                log(f"Reclaimed stale index lock from {writer_lock.reclaimed_from}")

        store = get_store(config, collection_name)
        if not args.dry_run:
            store.ensure(collection_name)

        existing_count = int(store.count())
        log(f"Existing chunks in index: {existing_count}")
        if total_files == 0 and existing_count:
            raise RuntimeError(
                f"Every source (markdown/PDF/JSON) reported 0 files, but the index "
                f"already holds {existing_count} chunks. Refusing to continue — "
                "check source mounts and run rag-config --strict."
            )

        with ReconciliationCatalog.from_store(store, index_path) as reconciliation:
            log(
                f"Reconciliation catalog loaded {reconciliation.existing_count} chunks "
                "using paged metadata"
            )
            persisted_state = read_index_state(store)
            if args.dry_run:
                log(
                    "DRY RUN: no embeddings, metadata updates, deletions, or "
                    "manifests will be written"
                )
                model = None
                persisted_dimension = (
                    persisted_state.provenance.embedding_dimension
                    if persisted_state is not None
                    else int(config.get("embedding_dimension", 0))
                )
                provenance = expected_provenance(
                    config,
                    collection_name=collection_name,
                    dimension=persisted_dimension,
                )
                previous_state = ensure_compatible(store, provenance)
            else:
                log("Loading embedding model...")
                model = SentenceTransformer(model_name, device=device)
                log("Model loaded.")
                provenance = expected_provenance(
                    config, model=model, collection_name=collection_name
                )
                previous_state = ensure_compatible(
                    store,
                    provenance,
                    adopt_missing=args.adopt_index_provenance,
                )
                begin_generation(store, provenance)

            results = []
            for source in sources:
                result = run_source(
                    source,
                    reconciliation,
                    model,
                    device,
                    embed_batch,
                    store,
                    dry_run=args.dry_run,
                )
                results.append(result)

            plan = build_prune_plan(
                reconciliation,
                results,
                max_fraction=float(config.get("prune_max_fraction", 0.25)),
                max_chunks=int(config.get("prune_max_chunks", 10_000)),
                allow_large_prune=args.allow_large_prune,
                allow_empty_source_prune=args.allow_empty_source_prune,
                dry_run=args.dry_run,
            )
            _log_prune_plan(plan)

            removed = 0
            manifest_path = None
            if not args.dry_run:
                removed = apply_prune_plan(store, plan, reconciliation)
                changed = bool(
                    removed
                    or sum(result.new_chunks for result in results)
                    or sum(result.updated_chunks for result in results)
                )
                state = publish_generation(store, provenance, previous_state, changed)
                manifest_path = write_run_manifest(config, collection_name, plan)
                log(f"Applied {removed} source-scoped stale deletions")
                log(f"Index generation: {state.generation_id}")
                log(f"Run manifest: {manifest_path}")

    total_chunks = sum(result.total_chunks for result in results)
    total_new = sum(result.new_chunks for result in results)
    total_updated = sum(result.updated_chunks for result in results)
    degraded = sum(result.state not in {"completed"} for result in results)
    mode = "plan complete" if args.dry_run else "indexing complete"
    log(
        f"\n{mode}. Processed {total_chunks} chunks "
        f"({total_new} new, {total_updated} metadata-updated, "
        f"{removed} removed, {degraded} non-complete sources)."
    )


if __name__ == "__main__":
    main()
