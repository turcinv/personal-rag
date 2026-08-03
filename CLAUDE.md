# personal-rag

Local retrieval index for an Obsidian knowledge vault + a PDF book/resource library. Embeds content as vector chunks in ChromaDB and retrieves by semantic query. Also contains the full document extraction pipeline (formerly `doc-text-extractor`). **Production target: an NVIDIA Jetson Orin Nano** (runs in the `Dockerfile.jetson` container); also runs on macOS/x86 for development.

**Canonical vault** is the maintained git checkout at `~/Documents/personal_knowledge/Career Knowledge Base/` (the Google Drive copy at `mindmap/Career Knowledge Base/` has a stale, divergent history — do not index it). **Books/resources** are indexed from pre-extracted JSON produced by the `src/extractor/` pipeline (`~/Documents/knowledge-base-index/indexed/`), not by parsing PDFs live.

## What this project does

- Extracts text from PDF/EPUB/Markdown source documents (PyMuPDF + Tesseract OCR)
- Builds a classified, FTS-searchable SQLite index of all books/resources + vault notes
- Generates per-resource Obsidian note stubs and injects backlinks into Topic MOCs
- Indexes vault Markdown notes and document JSON into a local collection, reached
  through the pluggable `RetrievalStore` seam (`src/rag/store/`, ChromaDB today)
- Retrieves relevant chunks by semantic similarity query, optionally reordered by a
  cross-encoder reranker — all callers share one `query.search()`
- Serves retrieval (and triggers reindexing) over HTTP via a JWT-authenticated FastAPI
  backend (`rag-serve`) that loads the model/store/reranker/generator once at startup
- Optionally synthesizes grounded, cited answers over the retrieved chunks via the
  `POST /answer` endpoint (the `generation/` layer). This sits ABOVE `search()` and is
  orthogonal to the store — off by default; enabled per-profile with a `generation`
  config block + a provider API key in the env (Anthropic / OpenAI). See docs/api.md.
- Fully offline for indexing and retrieval — no cloud APIs used there. The optional
  `/answer` generation layer is the one part that calls out to a cloud LLM (and only
  when explicitly configured); retrieval never does.
- Compatible with both macOS/x86 and NVIDIA Jetson (Python 3.10, same codebase)

## Environment

**Always use the local `.venv`, never the system Python or Conda.**

```bash
# First-time setup
make install          # creates .venv and installs package + deps

# RAG indexing and querying
.venv/bin/rag-index
.venv/bin/rag-query "your question"
.venv/bin/rag-query "your question" -n 12    # optional: number of results (default 8)
.venv/bin/rag-query "Kubernetes" --domain DevOps

# Optional BM25+dense hybrid (experimental, off by default — see roadmap item 6).
# Build the lexical index once (from the existing collection; no re-embed), then --hybrid:
.venv/bin/rag-build-lexical                  # or: make build-lexical
.venv/bin/rag-query "your question" --hybrid

# HTTP backend (query + indexing over HTTP; JWT-authenticated). Full guide: docs/api.md
export RAG_API_JWT_SECRET="$(openssl rand -hex 32)"   # required; ≥32 bytes
make serve                                            # or: .venv/bin/rag-serve  (0.0.0.0:8000)
.venv/bin/rag-token --subject telegram-bot            # mint a service token for a client
make jetson-serve                                     # run the api compose service on the Jetson

# Extraction pipeline (run in order for a full pipeline run)
make extract          # extract text from Books + Resources dirs
make enrich           # enrich inventory metadata
make build-index      # join inventory + text into indexed/*.json
make build-notes      # generate Obsidian Resource Notes
make build-sqlite     # build FTS5 SQLite database
make build-vault-index  # index vault Knowledge/ notes into JSONL
make dup-detect       # near-duplicate detection report
make link-mocs        # inject resource backlinks into Topic MOCs
make search Q="..."   # CLI FTS search over resources.db

# Run offline unit tests (pytest, ~195 tests, no network/model/index needed).
# `make install` uses --no-deps so it does NOT install pytest — add it once:
#   uv pip install pytest
make test-unit                      # or: .venv/bin/python -m pytest tests/

# Run retrieval smoke tests (needs a populated index)
.venv/bin/python tests/test_queries.py
.venv/bin/python tests/test_queries.py kubernetes   # filter by keyword

# Measure retrieval quality (needs a populated index)
make eval                           # recall@5/@10 + MRR over tests/eval/golden_queries.jsonl
make eval ARGS=--rerank             # compare rerank modes on the same index

# Recreate environment from scratch
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

Do not use bare `python` or `python3` — the Conda base environment will be picked up instead of `.venv`.

## Project layout

```
personal-rag/
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── utils.py          # load_config, logging setup, telemetry suppression, .env
│   │   ├── chunking.py       # heading split, char chunking, stable IDs, wikilinks
│   │   ├── extractors/       # one module per source type + iter_sources() registry
│   │   │   ├── __init__.py    #   Source namedtuple + iter_sources(config)
│   │   │   ├── markdown.py    #   extract_md_file, should_exclude
│   │   │   ├── pdf.py         #   extract_pdf_file, clean_pdf_title
│   │   │   └── json_doc.py    #   extract_json_doc (pre-extracted indexed/*.json)
│   │   ├── indexing.py       # incremental engine: embed/upsert, per-file diff, run_source
│   │   ├── indexer.py        # main() orchestration (MD + PDF + JSON → store); 0-files guard; entry: rag-index
│   │   ├── query.py          # search()/build_where/rerank_default seam + query CLI; entry: rag-query
│   │   ├── lexical.py        # BM25 FTS5 lexical index + RRF fusion (hybrid=True path); entry: rag-build-lexical
│   │   ├── eval.py           # recall@5/@10 + MRR over tests/eval/golden_queries.jsonl (make eval)
│   │   ├── pipeline_status.py # extractor pipeline pre-flight check; entry: rag-pipeline-status
│   │   ├── store/            # pluggable backend seam (ADR Axis 2) — ONLY place chromadb is imported
│   │   │   ├── base.py        #   RetrievalStore Protocol (10 members incl. snapshot/update_metadata/iter_records + supports_hybrid)
│   │   │   ├── chroma_store.py #  ChromaStore — the one implementation
│   │   │   └── __init__.py    #   get_store(config, collection_name); rejects any store but "chroma"
│   │   ├── generation/       # answer-synthesis layer ABOVE search() (retrieval ≠ chatbot); orthogonal to store
│   │   │   ├── base.py        #   Generator Protocol + AnswerResult + shared build_prompt/format_contexts ([n] citations)
│   │   │   ├── anthropic_gen.py #  AnthropicGenerator (Messages API via httpx, no SDK)
│   │   │   ├── openai_gen.py  #   OpenAIGenerator (Chat Completions via httpx; also OpenAI-compat servers)
│   │   │   └── __init__.py    #   get_generator(config) factory (provider from generation.* config; key from env)
│   │   └── api/              # FastAPI HTTP backend (reuses query.search — never reimplements retrieval)
│   │       ├── app.py         #   app + lifespan: load model/collection/reranker + generator ONCE; entry: rag-serve
│   │       ├── auth.py        #   HS256 JWT bearer auth dependency (require_jwt); secret from RAG_API_JWT_SECRET
│   │       ├── token.py       #   mint a service JWT for the bots; entry: rag-token
│   │       ├── deps.py        #   get_rag_state (shared app.state accessor)
│   │       ├── jobs.py        #   background reindex subprocess manager + in-process job registry
│   │       ├── schemas.py     #   pydantic request/response models
│   │       └── routes/        #   query.py (/health, /query, /status) + answer.py (/answer) + index.py (/index, /index/jobs/{id})
│   └── extractor/            # document extraction pipeline (merged from doc-text-extractor)
│       ├── __init__.py
│       ├── extract_text.py    # PDF/EPUB/MD → text_output/*.json; entry: rag-extract
│       ├── analyze_files.py   # pre-flight survey of a dir; entry: rag-analyze
│       ├── enrich_metadata.py # enrich inventory from embedded fields + ISBNs; entry: rag-enrich
│       ├── build_index_documents.py  # join inventory + text; entry: rag-build-index
│       ├── build_obsidian_notes.py   # generate Resource Notes; entry: rag-build-notes
│       ├── build_books_index.py  # regenerate Resources/Generated/Books Index.md; entry: rag-build-books-index
│       ├── build_sqlite.py    # FTS5 SQLite database; entry: rag-build-sqlite
│       ├── build_vault_index.py  # vault notes → JSONL; entry: rag-build-vault-index
│       ├── dup_detect.py      # near-duplicate detection; entry: rag-dup-detect
│       ├── link_mocs.py       # inject backlinks into MOCs; entry: rag-link-mocs
│       └── search.py          # CLI FTS search; entry: rag-search
├── tests/                    # `make test-unit` runs everything here EXCEPT test_queries.py
│   ├── conftest.py           # collect_ignore = ["test_queries.py"] (needs a real index + model)
│   ├── test_chunking.py      # unit: chunking/ID helpers
│   ├── test_extractors.py    # unit: markdown/pdf/json extractors (rag package)
│   ├── test_extractor.py     # unit: extractor package (detect_type, merge, shingles, etc.)
│   ├── test_indexing.py      # unit: incremental engine + idempotency (fake model + temp chroma)
│   ├── test_store.py         # unit: ChromaStore parity, config profiles, chromadb-confinement guard
│   ├── test_query_search.py  # unit: search() seam — dense pool, rerank reorder/trim, fetch width, hybrid fusion
│   ├── test_lexical.py       # unit: BM25 FTS5 index + RRF fusion (temp sqlite, no chromadb/model)
│   ├── test_generation.py    # unit: generation layer (httpx.MockTransport, no network)
│   ├── test_eval.py          # unit: eval matching/aggregation + golden-set well-formedness
│   ├── test_api.py           # unit: FastAPI backend via TestClient + fake state (dependency_overrides)
│   ├── test_queries.py       # 13-query retrieval smoke test (needs a populated index; NOT collected)
│   └── eval/                 # golden_queries.jsonl (45 labelled queries) + recorded eval snapshots
├── scripts/
│   ├── eval_recall.py        # shim → rag.eval.main (make eval)
│   └── drop_collections.py   # drop rejected trial collections (bge/gte)
├── .github/workflows/ci.yml  # CI: runs the offline unit suite on push/PR (Python 3.10 only)
├── Dockerfile                # x86 / macOS container image (python:3.10-slim + tesseract)
├── Dockerfile.jetson         # Jetson JetPack 6.2 container image (build on Jetson)
├── docker-compose.yml        # x86 Compose with volume mounts (RAG + extractor paths)
├── docker-compose.jetson.yml # Jetson Compose (runtime: nvidia)
├── .venv/                    # local virtualenv — never commit
├── chroma_db/                # ChromaDB data — never commit
├── .env                      # path overrides — never commit (see .env.example)
├── .env.example              # template for .env
├── .dockerignore
├── config.yaml               # default config profile: vault path, pdf + json sources, model, chunk settings, extractor paths
├── config.personal.yaml      # config profile: personal KB (light/Jetson) — select via RAG_CONFIG_PATH
├── config.logmanager.yaml    # config profile: Logmanager wiki (heavy/x86, markdown-only) — select via RAG_CONFIG_PATH
├── pyproject.toml            # package definition and CLI entry points (rag + extractor)
├── Makefile                  # shortcuts: make install/index/query/test/extract/enrich/...
├── requirements-direct.txt   # direct deps only — use to regenerate lockfile
├── requirements.txt          # full pinned lockfile — x86 / macOS (compiled for Python 3.10)
├── requirements-jetson.txt   # pinned deps — Jetson JetPack 6.2 (aarch64, CUDA 12.6)
├── docs/
│   ├── architecture.md       # pipeline, chunk IDs, store seam, rerank, generation, telemetry
│   ├── configuration.md      # full config reference, env var overrides, config profiles
│   ├── jetson.md             # Jetson install, Docker, memory budget, GPU constraints
│   ├── api.md                # HTTP backend: endpoints, JWT, client examples
│   ├── ADR-multi-corpus-profiles-and-pluggable-store.md   # Axis 1 + 2 design decision
│   ├── OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md             # Axis 3 — planned, not built
│   ├── RAG_ITEM7_TAG_STATUS_FILTERS_PLAN.md               # completed, kept for history
│   └── archive/              # historical reports + completed plans — NOT current reference
├── README.md                 # setup and usage guide
└── CLAUDE.md                 # this file
```

## Key dependencies (pinned)

Declared in `requirements-direct.txt` (the source of truth) and mirrored in
`pyproject.toml`; `requirements.txt` is the compiled lockfile and
`requirements-jetson.txt` the hand-maintained aarch64 equivalent.

| Package | Version | Used for |
|---|---|---|
| chromadb | 0.6.3 | vector store behind `ChromaStore` |
| sentence-transformers | 3.3.1 | bi-encoder embedder **and** the CrossEncoder reranker |
| torch | 2.5.1 | installed separately per arch — see `pyproject.toml` header |
| python-frontmatter | 1.3.0 | vault note YAML frontmatter |
| PyYAML | 6.0.3 | config profiles |
| pypdf | 6.12.2 | live PDF parsing fallback (`extractors/pdf.py`) |
| PyMuPDF | >=1.24 | PDF/EPUB extraction, OCR page rendering (`src/extractor/`) |
| cryptography | >=3.1 | AES-encrypted PDF support |
| posthog | 7.17.0 | pinned only to keep the chromadb telemetry patch working |
| python-dotenv | 1.2.2 | `.env` path overrides |
| fastapi | 0.136.3 | HTTP backend (`rag-serve`) |
| uvicorn[standard] | 0.49.0 | ASGI server for `rag-serve` |
| pyjwt | 2.13.0 | HS256 bearer auth (`api/auth.py`) |
| httpx | 0.28.1 | provider REST calls in `generation/` (no provider SDK) |

**Tesseract is a system binary, not a Python package** — `src/extractor/extract_text.py`
shells out to it, so there is deliberately no `pytesseract` dependency. It is installed
in both Dockerfiles and in CI.

Regenerate the lockfile after touching `requirements-direct.txt`:

```bash
uv pip compile requirements-direct.txt --python-version 3.10 -o requirements.txt
```

`requirements-jetson.txt` is **not** generated — add new direct deps to it by hand,
pinned to whatever the x86 compile resolved.

## What gets indexed (and how) — current behavior

- **Vault Markdown** (~2,996 indexed notes of 3,363 `.md` in the vault): frontmatter → metadata. Before embedding, each note's
  body is stripped of the navigation tail (`# Related Topics` / `## Potential New Notes`) and
  inline `[[wikilink]]` syntax — the link graph still lands in the `wikilinks` metadata field
  (extracted from the full body first). See `chunking.strip_navigation_tail` / `strip_wikilink_syntax`.
- **Books/resources**: from `json_sources` (`indexed/*.json`) — full text + enriched metadata.
- **PDF sources are a fallback only**: any PDF whose `file_name` is covered by a JSON doc is
  **skipped** (dedup), so books index exactly once. Live PDF parsing only handles new files the
  extractor hasn't processed yet. (See `extractors.iter_sources` / `_json_covered_filenames`.)
  **Gotcha:** dedup matches the *full filename incl. extension*. A book present as both `x.epub`
  and `x.pdf` extracts to one `text_output/x.json` (stem collision, epub wins), so `indexed/`
  is epub-keyed and the `.pdf` twin counts as uncovered → live-parsed into a duplicate chunk set.
  Both the PDF glob and the extractor's listing are non-recursive, so park the redundant twin in
  a subdir (`Books/_superseded/`). Done 2026-07-27 for the two Java titles.
- **Excluded**: `Resources/Generated` (auto catalog stubs that double-cover JSON), `* MOC.md`
  (navigation-only wikilink indexes, via `exclude_filename_patterns`), plus Templates/Archive/etc.
- **Broken YAML frontmatter silently drops a note from the index.** An unquoted colon in a
  value (`title: Modbus Troubleshooting: Resolving …`) makes `frontmatter.loads` raise, the
  indexer logs one `SKIP <name>: mapping values are not allowed in this context` line among
  thousands, and the note ends up with **zero chunks** — it is unsearchable, not merely stale.
  Three notes were in this state until 2026-07-27. Audit with:
  `grep -c "SKIP .*mapping values" logs/rag.log` after a run; fix by quoting the value.

## ChromaDB state

- Collection: `obsidian_markdown` (model `all-MiniLM-L6-v2`, **202,132 chunks** as of
  2026-07-27 — 16,167 from 2,996 vault notes + 185,965 from 259 books/resources). The
  name encodes the model: chunk IDs hash chunk *text*, not the embedding, so a
  model swap must use a fresh collection or old vectors silently survive.
  `obsidian_markdown_bge_small` (`bge-small-en-v1.5`) and `obsidian_markdown_gte_small`
  (`thenlper/gte-small`) were both trialled and rejected (net regression vs MiniLM).
  See docs/architecture.md → "Changing the embedding model".
- Metadata fields: `path`, `title`, `heading`, `type`, `domain`, `subdomain`, `status`, `source`, `confidence`, `tags`, `wikilinks`.
  `subdomain` comes from frontmatter or the containing subfolder (`extractors/markdown.py`)
  and is filterable via `--subdomain` / `filters.subdomain`.
- Reindexing is incremental: chunk IDs are SHA-256 of the full chunk content, so unchanged chunks are skipped, changed/new ones embedded, and stale chunks (edited or deleted sources) pruned. The collection is never wiped. **Note:** the nav-tail/wikilink stripping changed chunk text, so the first run after those changes re-embeds the vault and prunes the old chunks (expected one-time churn).

## Config profiles

`config.yaml` is one **profile**; `config.personal.yaml` (personal KB, light/
Jetson) and `config.logmanager.yaml` (Logmanager wiki, heavy/x86, markdown-
only) are two more. `RAG_CONFIG_PATH` selects the profile per instance — no
code change (`load_config()` → `_find_config()` in `src/rag/utils.py` already
resolves it). Each config also sets `store: chroma` (default, forward-looking
for a future OpenSearch backend). Run two independent instances, one per
corpus, each with its own `RAG_CONFIG_PATH`:

```bash
RAG_CONFIG_PATH=./config.personal.yaml .venv/bin/rag-index
RAG_CONFIG_PATH=./config.logmanager.yaml .venv/bin/rag-serve
```

Full comparison table + two-instance run details: docs/configuration.md →
"Config profiles". Design rationale: docs/ADR-multi-corpus-profiles-and-pluggable-store.md.

## Deployment (Jetson Orin Nano)

- Run `make build-jetson` then `make jetson-index` **on the Jetson** (aarch64 PyTorch wheels can't build on x86). Compose mounts host paths via `.env`.
- **Two datasets must be synced to the Jetson** (the rest is optional fallback): the vault checkout (`git`/`rsync`) and `~/Documents/knowledge-base-index/indexed/` (`rsync`). PDF mounts (`RAG_PDF_BOOKS_PATH` / `RAG_PDF_RESOURCES_PATH`) are optional.
- **Gotcha:** Docker may create the mounted `indexed/` dir as **root** → rsync fails with "Permission denied". Fix on the Jetson: `sudo chown -R turcinv:turcinv ~/knowledge-base-index`.
- **Incident (2026-07-15):** the Jetson's `.env` was missing `RAG_VAULT_PATH`/`RAG_JSON_PATH` (only the extractor-pipeline vars were present), so Compose silently fell back to mounting the host's empty `/tmp` for `/vault`, `/books`, `/resources`, `/index`. `make jetson-index` saw 0 files in every source and pruned the *entire* existing collection (172,557 chunks → 0), since the incremental engine can't distinguish "mounts are broken" from "everything was really deleted." Root cause was `.env` never got all four indexer vars from `.env.example`, not a code bug. Recovered by fixing `.env` and re-running `make jetson-index` (full re-embed, since chunk IDs are content-hashed — no permanent data loss).
  - **Guard added in response:** `src/rag/indexer.py`'s `main()` now aborts with `RuntimeError` (no pruning) if every source reports 0 files while the index already holds chunks — this exact failure mode can no longer wipe the collection, it'll error out instead. Before every `make jetson-index`, `.env` on that host should have `RAG_VAULT_PATH` and `RAG_JSON_PATH` set (see `.env.example`) — verify with `docker compose -f docker-compose.jetson.yml config | grep -A2 volumes` if in doubt.
- Adding new books: run the extractor pipeline (`make extract → make enrich → make build-index`) then rsync `indexed/` to the Jetson and run `make jetson-index`.
- Full details in [docs/jetson.md](docs/jetson.md) ("Data on the Jetson" section).

## Adding new books (end-to-end)

```bash
# 1. Drop the PDF/EPUB into ~/Documents/personal_knowledge/Books/ or Resources/ (local disk —
#    GCS/Google Drive are no longer used as of 2026-07; disk is the single source of truth)
# 2. Add the classification record to Resources/_catalog/resource_inventory.jsonl
# 3. Run the extraction pipeline
make extract            # extract text
make enrich             # enrich metadata
make build-index        # produce indexed/*.json
make build-notes        # update Resource Notes in the vault
make build-books-index  # regenerate Resources/Generated/Books Index.md (else it drifts)
make build-sqlite       # update FTS database
make link-mocs          # update MOC backlinks

# 4. Reindex
make index          # or: make jetson-index if running on Jetson
```

Verify the inputs/outputs of each step first with `make pipeline-status`.

## Known Limitations & Improvement Roadmap (RAG quality review, 2026-07-13)

Independent review of retrieval quality (not code style) found the indexing pipeline
solid but the retrieval layer under-designed for this corpus. Full findings in
`Templates/RAG Quality Review Report.md` in the vault repo (Career Knowledge Base).
Summary, ranked by impact/effort — do these in order, and build the eval set (item 3)
*before* items 4-5 so changes can actually be measured:

1. **[DONE]** ~~Drop the triple L2/cosine/dot collection scheme.~~ Removed
   `compare.py` / `rag-compare`, the `--metric` indexer flag, and the
   `jetson-index-{all,cos,dot}` / `jetson-compare` Makefile targets. All embeddings
   are normalized, so cosine/dot/L2 rank identically — the triple scheme wasted ~2/3
   of Jetson build time/storage for zero quality gain. The indexer now always creates
   collections with `hnsw:space: cosine` (Chroma ignores this on a pre-existing
   collection, so no rebuild was triggered). One collection: `obsidian_markdown`.
2. **[DONE — MiniLM kept; both alternatives rejected]** Tried swapping
   `all-MiniLM-L6-v2` for a stronger 384-dim model (+ `chunk_max_chars` 1200 → 1000).
   Both candidates **net-regressed** vs the MiniLM baseline on this corpus (they
   helped resource queries but hurt the short vault-title queries that dominate):
   - **bge-small-en-v1.5** (`tests/eval/bge_small.json`): overall recall@5
     0.956→0.867, MRR 0.814→0.782; vault recall@5 0.933→0.80. Query-prefix
     implementation audited and confirmed correct (query-side only, fires, eval
     shares the path) — the regression is the model, not a bug.
   - **gte-small** (`tests/eval/gte_small.json`): also a net regression (no prefix
     needed, so not a prefix issue either).
   Reverted `config.yaml` to `all-MiniLM-L6-v2` / `obsidian_markdown` / chunk 1200.
   The trial collections should be dropped on the Jetson (see `scripts/drop_collections.py`).
3. **[DONE]** ~~Build a small labeled eval set + recall@k.~~ `tests/eval/golden_queries.jsonl`
   (45 real queries, 30 vault + 15 resource) + `src/rag/eval.py` (recall@5/@10, MRR,
   per-corpus) via `make eval` / `make jetson-eval`. Baseline recorded in
   `tests/eval/baseline.json`.
4. **[DONE]** ~~Heading/section-aware chunking for books & resources.~~
   `json_doc.py` now splits on `##`+ headings (`split_by_headings(min_level=2)` —
   a lone `#` is code-comment noise in book PDFs) and packs each section on
   paragraph/sentence boundaries via `chunking.chunk_paragraphs` (never mid-
   sentence); the real heading replaces the old `part N`. Takes effect on the
   next reindex (book/resource chunk IDs change → incremental re-embed + prune).
5. **[DONE — default is now per-profile and measured]** ~~Add a cross-encoder rerank
   step.~~ `query.py` `search()` retrieves `rerank_fetch_k` (20) dense candidates and
   reorders to the top `n_results` with `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   Query-time only, no reindex. Config: `reranker_model`, `rerank_fetch_k`,
   **`rerank_default`**.

   `search()` always defaults to `rerank=False`; what the *callers* do when the caller
   said nothing is resolved by `query.rerank_default(config)` — one seam shared by
   `rag-query`, `make eval`, `POST /query` and `POST /answer` (absent key ⇒ `True`, the
   pre-2026-07-27 behaviour). Per call: `rag-query --rerank` / `--no-rerank`, or
   `"rerank": true|false` in the request body (omit the field to take the profile default).
   **`config.yaml` / `config.personal.yaml` set `false`; `config.logmanager.yaml` keeps
   `true`** — the finding below is corpus-specific and there is no wiki eval set yet.

   **Measured 2026-07-27 on the 45-query golden set (202,132 chunks), both modes same run:**

   | | rerank OFF | rerank ON |
   |---|---|---|
   | overall recall@5 | **0.911** | 0.844 |
   | overall recall@10 | **0.956** | 0.933 |
   | vault recall@5 | **0.900** | 0.767 |
   | resource recall@5 | 0.933 | **1.000** |
   | resource MRR | 0.910 | **1.000** |

   Per-query: rerank moved 7 queries up and 7 down, but asymmetrically — six of the seven
   wins are reshuffles *inside* top-5 (4→3, 2→1, 3→2) that only lift MRR, while the losses
   eject correct hits from top-5 entirely (1→8, 2→8, 1→6, 2→7, 9→miss). Net −3 queries at
   k=5. The one substantive win is a resource query (7→1), which is what takes resource
   recall@5 to 1.000. **On this corpus rerank trades vault recall for resource ordering and
   loses overall.** Snapshots: `tests/eval/post_update_{rerank,norerank}.json`. Flipping the
   default is a live behaviour change for `rag-query` and the `/query` + `/answer` endpoints,
   so it is left as an open decision — note the eval only scores retrieval, not `/answer`
   synthesis quality, which rerank may still help.
6. **[DONE — tested on Chroma; net recall-neutral, off by default]** BM25/lexical
   hybrid path. **On the OpenSearch backend this is free** — the native
   `hybrid=True` path (see `docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md`, Axis 3).
   That doc *asserted* Chroma "doesn't need it"; this item **tested that hypothesis**
   instead of assuming it. Built a client-side BM25 index and measured.

   **Implementation (2026-07-30):** `search()` now has a real `hybrid=True` path
   (`src/rag/query.py`) — it fuses a dense pool with a BM25 pool via Reciprocal
   Rank Fusion (RRF, keyed by document text, deterministic tiebreak). The BM25 pool
   comes from a SQLite **FTS5** lexical index (`src/rag/lexical.py`, `porter
   unicode61`, stdlib — no new dep, works on Jetson), built by `rag-build-lexical`
   / **`make build-lexical`** from the *existing* collection via the store's new
   paged `iter_records()` — no re-embed, no re-chunk (14 s for 202,132 chunks →
   `./lexical_index/obsidian_markdown.db`, gitignored). Chroma sets
   `supports_hybrid = False`, so fusion is client-side, ABOVE the store; a future
   OpenSearchStore would fuse natively and `search()` would defer to it. Config:
   `hybrid_fetch_k` (50, pool depth both sides), `hybrid_weights` ([w_lexical,
   w_dense]), `hybrid_rrf_k` (60). Per call: `rag-query --hybrid`,
   `make eval ARGS="--hybrid"`. **Off by default** — `hybrid=False` is byte-identical
   to dense-only (proven: eval unchanged from `post_update_norerank.json`).

   **Measured 2026-07-30 on the 45-query golden set (202,132 chunks), `--no-rerank`:**

   | | dense (baseline) | hybrid `[1,1]` | hybrid `[0.3,1.0]` | hybrid+rerank `[1,1]` |
   |---|---|---|---|---|
   | overall recall@5 | **0.911** | 0.867 | **0.911** | 0.800 |
   | overall recall@10 | **0.956** | 0.911 | **0.956** | 0.889 |
   | overall MRR | 0.794 | 0.771 | **0.820** | 0.724 |
   | vault recall@5 | **0.900** | 0.800 | 0.867 | 0.700 |
   | resource recall@5 | 0.933 | **1.000** | **1.000** | **1.000** |
   | resource MRR | 0.910 | 1.000 | **1.000** | 0.933 |

   **Finding: the plan doc's "Chroma doesn't need hybrid" holds for *recall*, but
   is too strong.** Equal weights `[1,1]` over-weight BM25 on short vault-title
   queries and **net-lose** (recall@5 0.911→0.867), same failure shape as the rerank
   experiment (item 5). But **dense-favored `[0.3,1.0]` is recall-neutral**
   (recall@5/@10 identical to dense) **and lifts MRR 0.794→0.820** — the gain is
   entirely in resources (all 15 resource queries rank their target #1; resource MRR
   0.910→1.000). Per-query, `[0.3,1.0]` moved 11 queries; the only k=5 crossings are
   one vault rank-1 hit dropping to 8 (−1 vault) offset by a resource straggler
   7→1 (+1 resource), netting zero at recall@5. Stacking rerank on top makes it
   worse (0.800). The one vault recall@5 loss is structural — it persists at every
   weight tried ([0.2,1.0], [0.3,1.0]) — so it can't be tuned away without giving
   back the resource win.

   **Decision: `hybrid` stays defaulting to `False`.** There is no recall gain to
   justify the extra build artifact + query cost by default; the improvement is
   ranking-only (MRR) and concentrated in resource queries. But it's a legitimate
   **opt-in** for resource-heavy query loads (perfect resource ranking at recall
   parity), and the shipped `hybrid_weights` default is `[0.3,1.0]` so anyone
   flipping `--hybrid` on gets the good operating point, not the naive equal-weight
   loss. Snapshot: `tests/eval/post_update_hybrid.json` (the `[0.3,1.0]` run).
   **Caveat: n=45 is underpowered** (±2.2%/query; the verdict rests on ~2–4 flips)
   and the set is 100% English + unfiltered, so this did **not** measure the
   Czech-keyword or filtered scenarios where lexical fusion most plausibly helps —
   the strongest case for hybrid on this corpus is still unmeasured.
7. **[DONE]** ~~Expose `--tag`/`--status` filters in `query.py`.~~ `status` is a
   native Chroma `$eq` clause in `build_where()`; `tags` (comma-joined string, weak
   Chroma array support) is a case-insensitive AND post-filter behind a widened
   `tag_fetch_k` (200) candidate pool — ranking unchanged when no filter is given.
   Wired through the CLI (`--status`, repeatable `--tag`) and the HTTP API
   (`QueryFilters.status`/`.tags` → `/query`). See
   `docs/RAG_ITEM7_TAG_STATUS_FILTERS_PLAN.md` for the implementation notes
   (superseded by the merged code; kept for history).

## Deeper reference

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline walkthrough, chunk IDs, store seam, anti-wipe guard, rerank, generation layer, telemetry workaround |
| [docs/configuration.md](docs/configuration.md) | All config fields, env var overrides, hardware tuning table, config profiles |
| [docs/jetson.md](docs/jetson.md) | Jetson-specific install, Docker, memory budget, IPC constraints, reranker cache prep |
| [docs/api.md](docs/api.md) | HTTP backend: endpoints, JWT auth, `rag-serve`/`rag-token`, `make serve`/`jetson-serve`, client examples |
| [docs/ADR-multi-corpus-profiles-and-pluggable-store.md](docs/ADR-multi-corpus-profiles-and-pluggable-store.md) | Design decision behind config profiles (Axis 1) + the pluggable `RetrievalStore` (Axis 2) |
| [docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md](docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md) | Axis 3 — planned OpenSearch backend, where roadmap item 6 (hybrid BM25) lands. Not built |
| [docs/archive/](docs/archive/) | Historical reports and completed plans. **Not current reference** — several contradict the code; see its README |

## Known drift and deferred work (audited 2026-07-28)

Found during a full repo review, deliberately not fixed:

- **`src/rag/api/jobs.py`** — the reindex job registry is never pruned; one record +
  thread ref per reindex lives for the server process's lifetime. The only `TODO` in
  `src/`. Harmless in practice (reindexes are rare).
- **There is no `rag-eval` console script**, unlike the other 16 `rag-*` entry points.
  The eval harness runs via `make eval` (→ `scripts/eval_recall.py`) or
  `python -m rag.eval`. Stale references to `rag-eval` as a command were corrected on
  2026-07-28; adding the entry point would be the tidier fix.
- **`make install` cannot run `make test-unit`** — `pytest` is only in the `dev` extra,
  which `--no-deps` skips. `uv pip install pytest` separately.
- **CI covers Python 3.10 only**, not the 3.12 used for macOS development, and has no
  lint or type-check step. `ci.yml`'s comment claims a `<3.11` pin that doesn't exist.
- **Config profiles don't work in Docker** — both images copy only `config.yaml` and
  neither Compose file passes `RAG_CONFIG_PATH`. Axis 1 is host-venv-only.
- **`tests/eval/baseline.json` is stale** (205,476 chunks, 2026-07-20). Compare against
  `post_update_norerank.json` (202,132) or refresh it.
- **`config.logmanager.yaml`'s `vault_path` doesn't exist yet** (declared placeholder).
  Indexing that profile creates an empty `wiki_lm` collection without erroring — the
  anti-wipe guard can't fire on an already-empty index.
