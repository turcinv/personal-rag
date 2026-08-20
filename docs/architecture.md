# Architecture

## Pipeline overview

```
Obsidian vault (.md)  ─┐
PDF books & resources  ─┼─► Per-file text extraction (ThreadPoolExecutor)
Pre-extracted JSON     ─┘         │
                                   │  one file at a time (streaming)
                                   ▼
                        Chunk by heading + character count
                        (1200 chars max, 150 overlap)
                                   │
                                   ▼
                        Embed with all-MiniLM-L6-v2
                        (single-process; CUDA on Jetson/GPU, CPU fallback)
                                   │
                                   ▼
                        Upsert batch → RetrievalStore → ChromaDB (./chroma_db)
                        free memory → next file
```

Retrieval reads back out through the same seam, and every retrieval caller shares
one function — `query.search()`:

```
   rag-query      POST /query     POST /answer      rag.eval
       └───────────────┴────────────────┴───────────────┘
                             │
                    query.search()
                             │
                  ┌──────────┴───────────┐
                  │  RetrievalFilter →   │  neutral filters: domain, subdomain,
                  │  RetrievalStore.query│  type, source, confidence, status
                  │  (store compiles)    │  (store builds backend syntax)
                  └──────────┬───────────┘
                             │  fetch_k candidates (widened for rerank/tags)
                             ▼
                  tag post-filter (case-insensitive AND)
                             │
                             ▼
                  optional cross-encoder rerank → top n_results
                             │
                             ▼
                  optional LLM synthesis  ← /answer only, off by default
```

The three layers are deliberately stacked, not merged: a **store** never ranks, a
**reranker** never generates, and `search()` never knows which caller invoked it.

## Streaming indexer

The indexer is split by responsibility:

| Module | Role |
|---|---|
| `chunking.py` | Pure text helpers — heading split, char-window chunking, content-hashed IDs, wikilinks |
| `extractors/` | One module per source type (`markdown`, `pdf`, `json_doc`), each returning the common `(ids, docs, metas, err)` tuple; `iter_sources(config)` turns config into a uniform stream of `Source`s |
| `indexing.py` | The incremental engine — `embed_and_upsert`, `index_file_chunks` (new/update/skip), `preserve_existing`, and `run_source()` |
| `indexer.py` | `main()` only — sets up the model/store, snapshots the index, runs every source, prunes stale chunks |
| `store/` | The `RetrievalStore` Protocol + `ChromaStore`; the **only** place a backend client is imported |
| `query.py` | `search()` — the single retrieval seam shared by the CLI, the API and the eval harness — plus `rerank_default()` and the CLI. The backend-neutral filter DTO lives in `retrieval.py`; the store compiles it |
| `eval.py` | recall@5/@10 and MRR over `tests/eval/golden_queries.jsonl`, split per corpus |
| `generation/` | Optional answer synthesis *above* `search()` — see [Answer generation](#answer-generation) |
| `api/` | FastAPI backend — see [HTTP backend](#http-backend) |

Adding a new source type is a new `extractors/<type>.py` + a block in `iter_sources` + a config entry — `main()` is untouched.

The pipeline is **fully streaming** — no global accumulation of chunks in RAM. Each file is processed end-to-end (extract → embed → upsert → free) before the next file starts.

### Per-file loop

1. **Extract** — I/O runs in a `ThreadPoolExecutor` (`markdown_workers` or `pdf_workers` threads):
   - Markdown: reads `.md` via `rglob`, skips excluded dirs/files, strips `{{...}}` Obsidian template vars, splits by heading, chunks by char count with overlap
   - PDF: reads pages into text blocks, chunks by char count; prefers embedded PDF title metadata over filename
   - JSON (`json_sources`): reads pre-extracted `indexed/*.json` (produced by this repo's own extraction pipeline, `src/extractor/`) — uses the document's full `text` and maps its metadata (`primary_topic`→`domain`, `resource_type`→`type`, plus `tags`/`confidence`/`title`). Splits on `##`+ headings and packs each section on paragraph/sentence boundaries, so a chunk never breaks mid-sentence and carries its real heading. Deterministic input, so re-runs are idempotent (unlike live PDF parsing)

2. **Embed** — always runs in the main thread (never multiprocess); `model.encode()` called with `embedding_batch_size` chunks at a time; `torch.cuda.empty_cache()` after each batch on CUDA

3. **Upsert** — each batch is written through `RetrievalStore.upsert()` immediately after embedding; no full-vault buffer

4. **Free** — `del ids/docs/metas` + `gc.collect()` after each file; `torch.cuda.empty_cache()` after each PDF

### Why single-process embedding

`encode_multi_process()` is intentionally avoided — Jetson uses NvSCI IPC (not CUDA IPC), so cross-process tensor sharing fails. Single-process GPU encoding is the correct path for both Jetson and desktop GPUs.

### Stable chunk IDs

SHA-256 of `(path, section_index, chunk_index, chunk)` — the full chunk text is hashed, so identical content always yields the same ID and any edit yields a new one. This is what makes incremental indexing reliable.

## The store seam (`src/rag/store/`)

Nothing in the indexing or retrieval engine talks to a vector database directly.
`store/base.py` defines a `RetrievalStore` Protocol and `store/chroma_store.py`
is the single implementation; `import chromadb` appears **only** there, and a unit
test in `tests/test_store.py` enforces that confinement.

The Protocol has nine members: `name`, `ensure`, `snapshot`, `existing_ids`,
`count`, `upsert`, `update_metadata`, `delete`, `query`. Two of them exist for
reasons worth knowing:

- **`update_metadata` is separate from `upsert`** so a metadata-only edit (a
  changed frontmatter field on an unchanged chunk body) can refresh metadata
  *without* recomputing the embedding. Folding it into `upsert` would silently
  make every frontmatter tweak a re-embed.
- **`snapshot` returns `{chunk_id: metadata}`, not just IDs**, because the
  incremental diff has to classify each candidate as new / metadata-changed /
  unchanged in one round-trip. `existing_ids` alone cannot express that.

`query()` returns records best-first and is explicitly forbidden from reranking —
rank assignment and the cross-encoder both live above it, in `search()`.

The backend is selected per profile by the `store:` config key via
`get_store(config, collection_name)`. Only `chroma` is accepted today; anything
else raises. `dim` is accepted by `ensure()` for interface parity but Chroma
ignores it (it infers dimensionality from the first upsert). Design rationale:
[ADR](ADR-multi-corpus-profiles-and-pluggable-store.md) Axis 2. A planned
OpenSearch backend — which is where hybrid BM25 retrieval would land — is specced
in [OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md](OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md)
but not built.

### Collection lifecycle (incremental)

The collection is opened with `get_or_create_collection` — never wiped. Each run:

1. Pages existing IDs and metadata into a compact temporary SQLite reconciliation catalog before any mutations. The backend page and SQLite insertion buffers are fixed-size; no corpus-sized Python dictionary or ID set is retained.
2. Submits at most the configured source worker count as extraction futures. Each completed file is classified against SQLite, embedded/upserted in small batches, and released before replacement work is queued. For every chunk:
   - **new ID** → embed + upsert;
   - **existing ID, metadata changed** → `collection.update` refreshes the metadata *without re-embedding* (embeddings depend only on the body, so a frontmatter or heading edit is cheap);
   - **existing ID, metadata identical** → skip.
3. If a file's extraction *fails* or unexpectedly returns no chunks, its already-indexed chunks are marked "seen" so a transient error never deletes good data; that source is marked degraded.
4. Builds a source-scoped prune plan. Only sources that completed cleanly may delete their own unseen IDs. Missing, unreadable, disabled, degraded, unexpectedly empty, removed-from-config, and legacy/unowned chunks are preserved. Large deletions are blocked for review.

Re-running on an unchanged vault embeds nothing. Editing only a note's frontmatter/heading updates metadata with no re-embedding. Changing `chunk_max_chars`/`chunk_overlap_chars` changes every chunk's text and therefore every ID, so the next run proposes re-embedding and pruning the old chunks; deletion thresholds may require `--allow-large-prune` after review.

### Source-scoped pruning and the 0-files backstop

A misconfigured `.env` on the Jetson once made every source report 0 files, and the
old global prune emptied the collection (172,557 chunks → 0). The indexer now records
`source_id`, `source_kind`, and `file_id` ownership on every successfully seen chunk.
Missing, unreadable, disabled, degraded, and unexpectedly empty sources preserve
their ownership while healthy sources reconcile independently. Legacy chunks without
ownership are also preserved until successfully seen again.

`indexer.py:main()` additionally raises `RuntimeError` if every source reports zero
files while an existing index contains chunks. Preview decisions with
`rag-index --dry-run`; completed mutating runs atomically write a per-source manifest.
Deletion thresholds (`prune_max_fraction` and `prune_max_chunks`) require
`--allow-large-prune` after review, and an intentional empty source also requires
`--allow-empty-source-prune`. Never use those overrides to bypass an unexplained
mount issue. An already-empty index has nothing for either guard to preserve.

### Index provenance and model changes

Chunk IDs hash chunk **text**, not embedding vectors, so model or chunker changes
cannot be inferred from IDs. Every collection now persists a fingerprint covering
the embedding model/revision/dimension, normalization, chunker version and settings,
metric, corpus/profile identity, and provenance schema. `rag-index`, `rag-query`, the
API, MCP, and eval validate that fingerprint before incremental writes or serving;
a mismatch fails with the differing fields instead of mixing vectors.

Use a new collection name for a model/chunker migration so the old generation remains
available for rollback. A legacy non-empty collection without provenance is rejected.
After independently verifying exactly how it was built, it can be migrated once with
`rag-index --adopt-index-provenance`; rebuilding into a new collection is safer.
Indexer mutations advance a generation identifier before writes, so an interrupted run
also makes the prior lexical sidecar stale. A successful no-op restores the previous
generation; successful inserts, metadata changes, or deletes publish the new one.
`rag-build-lexical` records that generation and provenance fingerprint, and hybrid
retrieval refuses a missing or stale sidecar with an actionable rebuild error.

Model history on this corpus (measured with the recall@k harness, `tests/eval/`): `obsidian_markdown` = `all-MiniLM-L6-v2` — **the shipped default**; `obsidian_markdown_bge_small` = `BAAI/bge-small-en-v1.5` and `obsidian_markdown_gte_small` = `thenlper/gte-small` were both tried and both **net-regressed** vs MiniLM (bge/gte helped resources but hurt the short vault-title queries that dominate the corpus) — not shipped. The prefix implementation was audited and confirmed correct, so the regression is the models themselves on this corpus, not a bug. MiniLM kept.

Query-side model prefixes: some retrieval models expect a short instruction prepended to the **query** (not to indexed passages) — bge-small-en-v1.5 used `"Represent this sentence for searching relevant passages: "`. This is applied via `config.yaml`'s `query_instruction`, prepended in `query.py`'s `search()`; it is **empty** for models that need no prefix (all-MiniLM, gte-small). The index/passage side never prefixes.

## Query (`src/rag/query.py`)

`search()` is the one retrieval path in the codebase. The CLI, `POST /query`,
`POST /answer` and the eval harness all call it; none of them reimplements
retrieval. It:

1. Embeds the query string with the same model (CPU — no device selection needed for a single inference), prepending `query_instruction` if the profile sets one
2. Takes a backend-neutral `RetrievalFilter` (`src/rag/retrieval.py`) carrying `domain`, `subdomain`, `type`, `source`, `confidence`, `status` and `tags`; the store compiles the scalar constraints into its own filter syntax (`rag.store.compile_where` builds Chroma's `$eq`/`$and` — the one place that syntax exists). Callers never construct store syntax. The legacy `build_where()` is a thin compat shim over the same compiler.
3. Decides the candidate pool width, which is the part worth understanding:
   - plain: `n_results`
   - reranking: `rerank_fetch_k` (default 20)
   - tag-filtered: `tag_fetch_k` (default 200)
4. Calls `RetrievalStore.query()` for that many candidates
5. Applies the tag post-filter, if any (see below)
6. Reranks, if enabled, and trims to `n_results`
7. Returns records; the CLI prints them or dumps JSON (`--json`)

### Why tags are a post-filter

Callers build a `RetrievalFilter` and never touch backend syntax; the store
compiles it. The same filter through the CLI or `POST /query` compiles to an
identical backend query. **Caller surfaces differ deliberately:** the CLI and the
eval harness expose every filter plus `--hybrid`; `POST /query` and `POST /answer`
expose the metadata filters and tags but not hybrid (dense-only, for predictable
latency and because hybrid needs a built lexical sidecar); the MCP `search` tool
is query-only (no filters, no hybrid) so host clients pass free-text questions.

`status` is a native store filter. `tags` is not: tags are stored as a single
comma-joined string because Chroma's array support is weak, so they cannot be
matched with a native clause. Instead `search()` widens the candidate pool to
`tag_fetch_k` and filters in Python — exact match per tag, case-insensitive,
multiple tags ANDed. Exact means `ci` does not match `ci-cd`.

Ranking is unchanged when no tag filter is given: the widened fetch only happens
when tags are actually supplied.

### Cross-encoder reranking

`search()` can reorder its dense candidates with
`cross-encoder/ms-marco-MiniLM-L-6-v2` (`reranker_model`). This is a **query-time**
change only — it needs no reindex, and toggling it never invalidates the index.

Resolution of the default is a deliberate three-level design:

- `search(rerank=...)` **always defaults to `False`** — the library-level function
  makes no policy decision.
- Callers that were given no explicit preference consult
  `query.rerank_default(config)`, the one shared seam. An **absent** `rerank_default`
  key resolves to `True`, preserving pre-2026-07-27 behaviour.
- A per-call override always wins: `--rerank` / `--no-rerank`, or
  `"rerank": true|false` in an API request body.

**The default is per profile because the result is corpus-dependent.** Measured on
the 45-query golden set, reranking is a net *loss* on the personal corpus: it lifts
resource recall@5 to 1.000 but drops vault recall@5 from 0.900 to 0.767, because its
wins are mostly reshuffles inside the top 5 (which only help MRR) while its losses
eject correct hits out of the top 5 entirely. So `config.yaml` and
`config.personal.yaml` set `false`, and `config.logmanager.yaml` keeps `true` (a
different corpus, and no wiki eval set exists yet). Full numbers: CLAUDE.md roadmap
item 5.

Note the eval harness scores *retrieval* only. Reranking may still improve
`/answer` synthesis quality, which nothing here measures.

## Answer generation

`src/rag/generation/` sits **above** `search()` and is orthogonal to the store —
the same generator works with any backend. This keeps a clean separation:
retrieval is not a chatbot, and the generator cannot influence ranking.

| Module | Role |
|---|---|
| `base.py` | `Generator` Protocol, `AnswerResult`, and the shared `build_prompt` / `format_contexts` that number contexts for `[n]` citations |
| `anthropic_gen.py` | Anthropic Messages API over `httpx` — no provider SDK |
| `openai_gen.py` | OpenAI Chat Completions over `httpx`; also works against OpenAI-compatible servers via `base_url` |
| `__init__.py` | `get_generator(config)` factory — provider from the `generation` config block, API key from the environment |

**Off by default.** With no `generation` block, or no API key exported,
`get_generator()` yields nothing, `/answer` returns 503, and `/query` is
unaffected. The API key is read from the env var named by `api_key_env` and is
never stored in a config file.

This is the only outbound network call in the project. Indexing and retrieval
remain fully offline.

## HTTP backend

`src/rag/api/` wraps the same seams for network clients (the Telegram and Wiki
bots) so they don't pay the CLI's per-query cold start.

| Module | Role |
|---|---|
| `app.py` | App + lifespan: loads model, store, reranker and generator **once** at startup; entry `rag-serve` |
| `auth.py` | HS256 JWT bearer dependency; secret from `RAG_API_JWT_SECRET` |
| `token.py` | Mints service tokens; entry `rag-token` |
| `jobs.py` | Background reindex subprocess manager + in-process job registry |
| `deps.py` / `schemas.py` | Shared state accessor, pydantic request/response models |
| `routes/` | `query.py` (`/health`, `/query`, `/status`), `answer.py` (`/answer`), `index.py` (`/index`, `/index/jobs/{id}`) |

Because indexing runs as a **subprocess**, a reindex triggered over HTTP cannot
take the server's model memory with it if it fails. The job registry is
in-process and is not pruned — one record per reindex persists for the life of the
server. Endpoint reference: [api.md](api.md).

## What a chunk record contains

`store.upsert(ids, embeddings, docs, metas)` writes four parallel arrays, so every
chunk in the collection persists four things:

| Part | What it is | Derived from |
|---|---|---|
| `id` | SHA-256 of `(path, section_index, chunk_index, chunk)` | the **full chunk text** |
| `embedding` | 384-dim float vector (`all-MiniLM-L6-v2`) | the **chunk body only** |
| `document` | the chunk text, stored **verbatim** | — |
| `metadata` | the 11 fields tabled below | frontmatter, path, PDF/JSON metadata |

Two consequences worth knowing:

**The store holds a full second copy of your text.** `document` is not a pointer —
retrieval returns the stored text, so nothing needs to re-read the vault or the PDFs
at query time. That is why `search()` can serve results with the source files
unmounted, and why the API container needs no source mounts to answer `/query`.

**The vector depends on the body, the metadata does not.** This asymmetry is not an
optimization detail — it is what makes the metadata-only refresh path *correct*.
Editing a note's frontmatter or heading changes its metadata but not its chunk body,
so the ID is unchanged and the existing embedding is still valid; `update_metadata()`
refreshes the fields without re-embedding. Fold that into `upsert` and every
frontmatter tweak silently becomes a re-embed.

### On-disk footprint

Measured 2026-07-28 at 202,132 chunks: **2.4 GB** total.

| Component | Size | Notes |
|---|---|---|
| `embedding_metadata_string_value` | 645 MB | the metadata values themselves |
| `embedding_fulltext_search_*` | ~730 MB | Chroma's FTS index over `document` |
| `embedding_metadata` (+ index) | 349 MB | 11 fields × 202k chunks ≈ 2.2M key-value rows |
| HNSW segment dir | 428 MB | the actual vector index |
| `embeddings` (+ index) | 61 MB | id ↔ vector bookkeeping in SQLite |

The distribution is counter-intuitive, so don't estimate from the vectors: raw text
is only ~66 MB (mean chunk is ~327 chars, well under `chunk_max_chars` because many
vault notes are short enough to be returned whole), and float32 vectors account for
~0.3 GB. **Metadata and the full-text index are the two largest costs.**

Note that ~730 MB of the store is an FTS index this project never queries — Chroma
builds it over every document automatically, and there is no lexical/BM25 retrieval
path on this backend (see roadmap item 6 in CLAUDE.md, which routes hybrid search to
the planned OpenSearch backend instead). Reducing the metadata field count is the
only lever here that would meaningfully shrink the store.

## Metadata per chunk

| Field | Source | Example |
|---|---|---|
| `path` | file path | `Knowledge/DevOps/K3s.md` |
| `title` | frontmatter or PDF metadata | `K3s Adoption Decision Framework` |
| `heading` | Markdown heading or page range | `Summary` / `p.12-15` |
| `type` | frontmatter or pdf_source type | `Knowledge`, `book`, `resource` |
| `domain` | frontmatter, or derived from the path | `DevOps` |
| `subdomain` | frontmatter, or derived from the subfolder | `Python & Backend Development` |
| `status` | frontmatter | `processed` |
| `source` | frontmatter or `pdf` | `ChatGPT`, `pdf` |
| `confidence` | frontmatter (empty for PDFs) | `high`, `medium` |
| `tags` | frontmatter, comma-joined | `kubernetes, containers` |
| `wikilinks` | extracted from body | `K3s, Docker` |

All of these except `tags` and `wikilinks` are usable as native filters; `tags` is
the post-filter described above.

## Store state

As of **2026-07-27**:

- Collection: `obsidian_markdown` — embedding model `all-MiniLM-L6-v2`,
  **202,132 chunks**
- **16,167** chunks from **2,996** vault notes + **185,965** from **259**
  books/resources
- `source: pdf` distinguishes PDF chunks from Markdown chunks
- `obsidian_markdown_bge_small` (bge-small-en-v1.5) and `obsidian_markdown_gte_small`
  (gte-small) were trialled in their own collections and rejected as net regressions
  — see "Index provenance and model changes" above. Drop them with
  `scripts/drop_collections.py`.

CLAUDE.md is the authoritative record of the current corpus; treat the numbers here
as a snapshot.

## Known issue: telemetry noise

ChromaDB 0.6.3 + posthog 7.x have a signature mismatch. `src/rag/utils.py` suppresses it at module import time:

```python
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import posthog as _posthog
_posthog.capture = lambda *a, **kw: None
```

Do not remove when upgrading chromadb until confirmed fixed.
