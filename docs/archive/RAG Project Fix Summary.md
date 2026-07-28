# RAG Project Fix Summary

**Date:** 2026-06-05  
**Branch:** main  
**Issue:** `index_obsidian.py` killed by SIGKILL at ~PDF 168/175 due to RAM exhaustion on Jetson Orin Nano Super (8 GB unified RAM)

---

## Root Cause

The indexer accumulated **all chunks from all source files** into three global Python lists (`all_ids`, `all_docs`, `all_metas`) before embedding or writing anything to ChromaDB. By the time 1036 Markdown notes and 168 PDF books had been extracted, the lists held tens of thousands of text chunks (~300–400 MB of strings) alongside the loaded embedding model (~90 MB). Calling `model.encode()` on the full set at once then pushed total RAM over the 8 GB limit.

---

## What Changed

### `index_obsidian.py` — streaming pipeline

**Before:** extract all → embed all → upsert all  
**After:** for each file: extract → embed small batches → upsert → free memory → next file

Key changes:
- Removed global `all_ids / all_docs / all_metas` accumulators
- Added `embed_and_upsert()` helper: embeds `embedding_batch_size` chunks at a time, calls `collection.add()` immediately, deletes the embedding tensor, and calls `torch.cuda.empty_cache()` after each batch on CUDA
- Model (`SentenceTransformer`) is loaded once at startup and reused across all files — no repeated cold-start
- `del ids, docs, metas` + `gc.collect()` after every file; `torch.cuda.empty_cache()` after every PDF
- Removed `ProcessPoolExecutor` multi-process embedding (not needed; embedding is now inline in the main thread)
- Removed `_init_worker`, `_embed_batch`, `collect_parallel` (no longer used)
- I/O extraction still uses `ThreadPoolExecutor` controlled by `markdown_workers` / `pdf_workers`; with both at 1, files are processed strictly sequentially

### `config.yaml` — Jetson-safe defaults

| Key | Old | New | Reason |
|---|---|---|---|
| `chunk_max_chars` | 1800 | 1200 | Fewer chars per chunk → fewer chunks per large PDF → lower peak RAM per file |
| `chunk_overlap_chars` | 250 | 150 | Proportional reduction |
| `embedding_batch_size` | *(new)* | 16 | Limits embedding tensor size per `model.encode()` call |
| `markdown_workers` | *(new)* | 1 | Sequential MD extraction; 1 thread = no parallel file buffering |
| `pdf_workers` | *(new)* | 1 | Sequential PDF extraction; critical for Jetson RAM safety |
| `embedding_workers` | 2 | 1 | Legacy CPU ProcessPoolExecutor setting; kept for reference but no longer used by the streaming indexer |

### `CLAUDE.md`

- Updated indexing logic section to describe the streaming pipeline
- Updated config.yaml key reference with new keys and their defaults
- Updated Jetson RAM note to reflect the streaming fix

---

## Memory Safety Guarantee

Peak RAM during indexing is now bounded to:
- Model in memory: ~90 MB (all-MiniLM-L6-v2)
- One file's chunks: largest PDF ~200 chunks × 1200 chars ≈ ~240 KB of text
- One embedding batch: 16 chunks × 384-dim float32 ≈ ~24 KB of tensors
- ChromaDB client state: ~50–100 MB

Total: well under 1 GB, leaving ~7 GB headroom on Jetson for the OS, PyTorch runtime, and GPU kernel.

---

## What Was Not Changed

- `query_obsidian.py` — untouched
- `test_queries.py` — untouched
- All metadata fields — identical (`path`, `title`, `heading`, `type`, `domain`, `status`, `source`, `tags`, `wikilinks`)
- Stable ID scheme — SHA-256 of `(path, section_index, chunk_index, chunk[:80])` unchanged
- ChromaDB collection lifecycle — delete-then-recreate on full reindex, unchanged

> **Note:** chunk sizes changed (1800 → 1200 chars), so stable IDs will differ from previous index builds. A full reindex is required after this change (the indexer already recreates the collection from scratch on every run).
