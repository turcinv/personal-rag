"""Streaming incremental indexer — Markdown + PDF + JSON → ChromaDB.

Orchestration only: load config, set up the model and collection, snapshot the
existing index, then run every configured source (see ``extractors.iter_sources``)
through the incremental engine (see ``indexing.run_source``) and prune stale
chunks. Entry point: ``rag-index``."""

import argparse
import logging
from pathlib import Path

from .utils import load_config, setup_logging  # sets telemetry env var and patches posthog before chromadb loads

import torch
from sentence_transformers import SentenceTransformer

from .extractors import iter_sources
from .indexing import run_source
from .store import get_store


logger = logging.getLogger("rag")
log = logger.info  # bound to the shared 'rag' logger; configured by setup_logging() in main()


def main():
    parser = argparse.ArgumentParser(description="Index vault + PDFs + JSON into ChromaDB.")
    parser.add_argument(
        "--collection", default=None, metavar="NAME",
        help="Override the collection name from config.yaml.",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    vault_path      = Path(config["vault_path"]).expanduser().resolve()
    collection_name = args.collection or config.get("collection_name", "obsidian_markdown")
    max_chars       = int(config.get("chunk_max_chars", 1200))
    overlap         = int(config.get("chunk_overlap_chars", 150))
    model_name      = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch     = int(config.get("embedding_batch_size", 16))
    md_workers      = int(config.get("markdown_workers", 1))
    pdf_workers     = int(config.get("pdf_workers", 1))

    if not vault_path.exists():
        raise RuntimeError(f"Vault path does not exist: {vault_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Vault: {vault_path}")
    log(f"Collection: {collection_name}  |  metric: cosine")
    log(f"Embedding model: {model_name}  |  device: {device}" +
        (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    log(f"Chunk max chars: {max_chars}  |  overlap: {overlap}")
    log(f"Embed batch size: {embed_batch}  |  md_workers: {md_workers}  |  pdf_workers: {pdf_workers}")

    log("Loading embedding model...")
    model = SentenceTransformer(model_name, device=device)
    log("Model loaded.")

    store = get_store(config, collection_name)
    # All embeddings are L2-normalized (indexing.py), so cosine/dot/L2 rank
    # identically — we standardize on a single cosine collection. Chroma ignores
    # this metadata for a collection that already exists, so pointing at a
    # pre-existing collection never forces a rebuild; only freshly created
    # collections get hnsw:space=cosine.
    store.ensure(collection_name)

    # Incremental indexing: snapshot the IDs + metadata already in the index.
    # Chunk IDs are content-hashed, so unchanged body text keeps the same ID;
    # each chunk is then embedded (new), metadata-refreshed (same body, changed
    # metadata), or skipped (identical). Every ID seen this run is recorded so
    # leftovers (edited or deleted sources) can be pruned at the end.
    existing_meta = store.snapshot()
    existing_ids = set(existing_meta)
    seen_ids = set()
    log(f"Existing chunks in index: {len(existing_ids)}")

    # Materialize sources up front so we can total up file counts across all of
    # them *before* embedding/pruning anything. If every source reports 0 files
    # while the index is non-empty, that's almost certainly a broken mount or a
    # misconfigured path (e.g. a missing env var silently falling back to an
    # empty dir) — not a real mass deletion. Abort instead of pruning the whole
    # collection. See incident 2026-07-15: a missing RAG_VAULT_PATH/RAG_JSON_PATH
    # in .env caused every source to resolve to an empty /tmp mount and wiped
    # 172k+ chunks in one run.
    sources = list(iter_sources(config, vault_path, max_chars, overlap))
    total_files_seen = sum(len(source.files) for source in sources)
    log(f"Total source files found: {total_files_seen}")
    if total_files_seen == 0 and existing_ids:
        raise RuntimeError(
            f"Every source (markdown/PDF/JSON) reported 0 files, but the index "
            f"already holds {len(existing_ids)} chunks. Refusing to prune — this "
            f"looks like a broken mount or misconfigured path, not a real deletion. "
            f"Check vault_path/pdf_sources/json_sources in config.yaml and any "
            f"RAG_VAULT_PATH/RAG_PDF_BOOKS_PATH/RAG_PDF_RESOURCES_PATH/RAG_JSON_PATH "
            f"env overrides before re-running."
        )

    total_chunks = 0     # chunks across successfully-extracted files this run
    total_new = 0        # chunks embedded this run
    total_updated = 0    # chunks whose metadata was refreshed (no re-embed)

    for source in sources:
        s_total, s_new, s_upd = run_source(
            source, existing_meta, seen_ids, model, device, embed_batch, store,
        )
        total_chunks  += s_total
        total_new     += s_new
        total_updated += s_upd

    # Prune chunks that no longer exist in any source (edited or deleted files).
    stale = existing_ids - seen_ids
    if stale:
        stale_list = list(stale)
        for i in range(0, len(stale_list), 500):
            store.delete(stale_list[i:i + 500])
        log(f"Removed {len(stale)} stale chunks (edited or deleted sources)")

    log(f"\nIndexing complete. Index now holds {len(seen_ids)} chunks "
        f"({total_new} embedded, {total_updated} metadata-updated, "
        f"{total_chunks - total_new - total_updated} unchanged, {len(stale)} removed).")


if __name__ == "__main__":
    main()
