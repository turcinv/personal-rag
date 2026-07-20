"""Incremental indexing engine.

Source-agnostic: given a ``Source`` (from ``extractors``) and the snapshot of
what's already in the collection, it embeds new chunks, refreshes metadata-only
changes without re-embedding, skips unchanged chunks, and records every ID seen
so the caller can prune stale ones. Works the same for Markdown, PDF, or JSON."""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor

import torch

logger = logging.getLogger("rag")
log = logger.info  # configured by setup_logging() in indexer.main()


def embed_and_upsert(model, device, docs, ids, metas, embed_batch_size, store):
    """Embed in small batches and upsert immediately; never accumulates in RAM."""
    n = len(docs)
    n_batches = (n + embed_batch_size - 1) // embed_batch_size
    for batch_idx, i in enumerate(range(0, n, embed_batch_size), 1):
        b_docs  = docs[i:i + embed_batch_size]
        b_ids   = ids[i:i + embed_batch_size]
        b_metas = metas[i:i + embed_batch_size]
        embeddings = model.encode(b_docs, normalize_embeddings=True, batch_size=embed_batch_size)
        store.upsert(ids=b_ids, embeddings=embeddings.tolist(), docs=b_docs, metas=b_metas)
        del embeddings
        if device == "cuda":
            torch.cuda.empty_cache()
        if n_batches > 1:
            log(f"      batch {batch_idx}/{n_batches}  ({min(i + embed_batch_size, n)}/{n} chunks)")


def index_file_chunks(ids, docs, metas, existing_meta, seen_ids,
                      model, device, embed_batch, store):
    """Incrementally index one file's chunks against the existing index.

    Records every chunk ID in ``seen_ids`` (used afterwards to prune stale
    chunks). Chunk IDs are content-hashed, so for each chunk:
      - new ID                       -> embed + upsert
      - existing ID, metadata changed -> refresh metadata only (no re-embed)
      - existing ID, metadata same    -> skip

    Embeddings depend only on the chunk body, so a metadata-only edit (e.g. a
    note's frontmatter or a heading) is applied with store.update_metadata
    without paying to re-embed. Returns (n_new, n_updated, n_total)."""
    seen_ids.update(ids)
    new_i = [k for k, cid in enumerate(ids) if cid not in existing_meta]
    upd_i = [k for k, cid in enumerate(ids)
             if cid in existing_meta and metas[k] != existing_meta[cid]]

    if new_i:
        embed_and_upsert(model, device,
                         [docs[k] for k in new_i], [ids[k] for k in new_i],
                         [metas[k] for k in new_i], embed_batch, store)
    if upd_i:
        store.update_metadata(ids=[ids[k] for k in upd_i],
                              metas=[metas[k] for k in upd_i])
    return len(new_i), len(upd_i), len(ids)


def preserve_existing(path_value, existing_meta, seen_ids):
    """Mark a source's already-indexed chunks as seen so the stale-prune step
    does not delete good data when that file's extraction fails this run."""
    seen_ids.update(cid for cid, m in existing_meta.items() if m.get("path") == path_value)


def _index_status(n_new, n_upd, n_total):
    if n_new or n_upd:
        return f"{n_new} new, {n_upd} meta / {n_total} chunks"
    return f"unchanged, {n_total} chunks"


def run_source(source, existing_meta, seen_ids, model, device, embed_batch, store):
    """Run one ``Source``'s files through the incremental engine.

    Returns (n_total, n_new, n_updated) for the source. Extraction runs in a
    thread pool; embedding/upsert happens here in the main thread."""
    log(f"\n{source.label}")
    if not source.files:
        return 0, 0, 0

    n = len(source.files)
    s_total = s_new = s_upd = 0
    with ThreadPoolExecutor(max_workers=source.workers) as pool:
        futures = [(pool.submit(source.extract, f), f) for f in source.files]
        for idx, (future, f) in enumerate(futures, 1):
            ids, docs, metas, err = future.result()
            if err:
                log(f"  [{idx}/{n}] SKIP {f.name}: {err.split(':', 1)[-1].strip()}")
                if source.preserve_key is not None:
                    preserve_existing(source.preserve_key(f), existing_meta, seen_ids)
                continue
            if not docs:
                continue
            n_new, n_upd, n_total = index_file_chunks(
                ids, docs, metas, existing_meta, seen_ids,
                model, device, embed_batch, store,
            )
            log(f"  [{idx}/{n}] {f.name}  ({_index_status(n_new, n_upd, n_total)})")
            s_total += n_total
            s_new   += n_new
            s_upd   += n_upd
            del ids, docs, metas
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    log(f"  complete: {s_total} chunks ({s_new} embedded, {s_upd} metadata-updated this run)")
    return s_total, s_new, s_upd
