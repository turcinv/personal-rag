# Architecture

## Pipeline overview

```
Obsidian vault (.md)  ─┐
PDF books & resources  ─┼─► Per-file text extraction (ThreadPoolExecutor)
Pre-extracted JSON     ─┘         │
                                   │  one file at a time (streaming)
                                   ▼
                        Chunk by heading + character count
                        (1000 chars max, 150 overlap)
                                   │
                                   ▼
                        Embed with BAAI/bge-small-en-v1.5
                        (single-process; CUDA on Jetson/GPU, CPU fallback)
                                   │
                                   ▼
                        Upsert batch → ChromaDB (local, ./chroma_db)
                        free memory → next file
                                   │
                          rag-query / python -m rag.query
                                   │
                                   ▼
                        Top-N chunks by cosine similarity
                        optional metadata filter (--domain / --type / --source / --confidence)
```

## Streaming indexer

The indexer is split by responsibility:

| Module | Role |
|---|---|
| `chunking.py` | Pure text helpers — heading split, char-window chunking, content-hashed IDs, wikilinks |
| `extractors/` | One module per source type (`markdown`, `pdf`, `json_doc`), each returning the common `(ids, docs, metas, err)` tuple; `iter_sources(config)` turns config into a uniform stream of `Source`s |
| `indexing.py` | The incremental engine — `embed_and_upsert`, `index_file_chunks` (new/update/skip), `preserve_existing`, and `run_source()` |
| `indexer.py` | `main()` only — sets up the model/collection, snapshots the index, runs every source, prunes stale chunks |

Adding a new source type is a new `extractors/<type>.py` + a block in `iter_sources` + a `config.yaml` entry — `main()` is untouched.

The pipeline is **fully streaming** — no global accumulation of chunks in RAM. Each file is processed end-to-end (extract → embed → upsert → free) before the next file starts.

### Per-file loop

1. **Extract** — I/O runs in a `ThreadPoolExecutor` (`markdown_workers` or `pdf_workers` threads):
   - Markdown: reads `.md` via `rglob`, skips excluded dirs/files, strips `{{...}}` Obsidian template vars, splits by heading, chunks by char count with overlap
   - PDF: reads pages into text blocks, chunks by char count; prefers embedded PDF title metadata over filename
   - JSON (`json_sources`): reads pre-extracted `indexed/*.json` (from the external `doc-text-extractor` pipeline) — uses the document's full `text` and maps its metadata (`primary_topic`→`domain`, `resource_type`→`type`, plus `tags`/`confidence`/`title`). Deterministic input, so re-runs are idempotent (unlike live PDF parsing)

2. **Embed** — always runs in the main thread (never multiprocess); `model.encode()` called with `embedding_batch_size` chunks at a time; `torch.cuda.empty_cache()` after each batch on CUDA

3. **Upsert** — each batch is written to ChromaDB immediately after embedding; no full-vault buffer

4. **Free** — `del ids/docs/metas` + `gc.collect()` after each file; `torch.cuda.empty_cache()` after each PDF

### Why single-process embedding

`encode_multi_process()` is intentionally avoided — Jetson uses NvSCI IPC (not CUDA IPC), so cross-process tensor sharing fails. Single-process GPU encoding is the correct path for both Jetson and desktop GPUs.

### Stable chunk IDs

SHA-256 of `(path, section_index, chunk_index, chunk)` — the full chunk text is hashed, so identical content always yields the same ID and any edit yields a new one. This is what makes incremental indexing reliable.

### Collection lifecycle (incremental)

The collection is opened with `get_or_create_collection` — never wiped. Each run:

1. Snapshots the IDs **and metadata** already in the collection.
2. For every chunk, records its ID as "seen" and then, since the ID is a hash of the body only:
   - **new ID** → embed + upsert;
   - **existing ID, metadata changed** → `collection.update` refreshes the metadata *without re-embedding* (embeddings depend only on the body, so a frontmatter or heading edit is cheap);
   - **existing ID, metadata identical** → skip.
3. If a file's extraction *fails* (parse/read error), its already-indexed chunks are marked "seen" so they are preserved — a transient error never deletes good data. Files that legitimately become empty are not preserved and are pruned.
4. After all sources are processed, deletes any indexed ID that was not seen this run — pruning chunks from edited files (old content) and from deleted files.

Re-running on an unchanged vault embeds nothing. Editing only a note's frontmatter/heading updates metadata with no re-embedding. Changing `chunk_max_chars`/`chunk_overlap_chars` changes every chunk's text and therefore every ID, so the next run re-embeds everything and prunes the old chunks — effectively a clean rebuild.

### Changing the embedding model → new collection name (required)

Chunk IDs hash the chunk **text**, not the embedding vector. Swapping `embedding_model` does **not** change any chunk's text, so the incremental engine would see the existing IDs and *skip* re-embedding — silently leaving the old model's vectors in place under those IDs. Combined with a `chunk_max_chars` change (which re-IDs only chunks long enough to be split — short notes are returned whole and keep their ID), the same collection would end up with a **mix** of old- and new-model vectors, which are not comparable.

The safe path, therefore, is a **new collection name that encodes the model**, set in `config.yaml`'s `collection_name`. A fresh (empty) collection has no existing IDs, so every chunk embeds from scratch with the new model; the old collection is left intact for rollback and can be deleted manually once the new one is validated. Convention: suffix the model, e.g. `obsidian_markdown_bge_small` for `BAAI/bge-small-en-v1.5` (was `obsidian_markdown` for `all-MiniLM-L6-v2`). Bump the suffix on every model change. (A documented full wipe-and-rebuild of the same collection would also work, but versioning the name is safer — it never risks a half-migrated collection and keeps the old vectors available to compare against.)

Query-side model prefixes: some retrieval models expect a short instruction prepended to the **query** (not to indexed passages). bge-small-en-v1.5 uses `"Represent this sentence for searching relevant passages: "`. This is applied via `config.yaml`'s `query_instruction`, prepended in `query.py`'s `search()`; leave it empty for models that need no prefix (all-MiniLM, gte-small). The index/passage side never prefixes.

## Query (`src/rag/query.py`)

1. Embeds the query string with the same model (CPU, no device selection needed for single inference)
2. Builds an optional ChromaDB `where` filter from `--domain`, `--type`, `--source` flags
3. Queries ChromaDB for top-N chunks by cosine similarity
4. Prints results or dumps as JSON (`--json`)

## Metadata per chunk

| Field | Source | Example |
|---|---|---|
| `path` | file path | `Knowledge/DevOps/K3s.md` |
| `title` | frontmatter or PDF metadata | `K3s Adoption Decision Framework` |
| `heading` | Markdown heading or page range | `Summary` / `p.12-15` |
| `type` | frontmatter or pdf_source type | `Knowledge`, `book`, `resource` |
| `domain` | frontmatter | `DevOps` |
| `status` | frontmatter | `processed` |
| `source` | frontmatter or `pdf` | `ChatGPT`, `pdf` |
| `confidence` | frontmatter (empty for PDFs) | `high`, `medium` |
| `tags` | frontmatter | `kubernetes, containers` |
| `wikilinks` | extracted from body | `K3s, Docker` |

## ChromaDB state

- Collection: `obsidian_markdown_bge_small` (embedding model `BAAI/bge-small-en-v1.5`;
  the name encodes the model — see "Changing the embedding model" above. Prior
  collection `obsidian_markdown` used `all-MiniLM-L6-v2`.)
- ~7,600 MD chunks + PDF chunks from 175 books and 35 resources (as of 2026-06-04)
- `source: pdf` distinguishes PDF chunks from Markdown chunks

## Known issue: telemetry noise

ChromaDB 0.6.3 + posthog 7.x have a signature mismatch. `src/rag/utils.py` suppresses it at module import time:

```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None
```

Do not remove when upgrading chromadb until confirmed fixed.
