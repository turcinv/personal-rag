# personal-rag

[![tests](https://github.com/turcinv/personal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/turcinv/personal-rag/actions/workflows/ci.yml)

Local semantic search over an Obsidian knowledge vault and PDF book library. Indexes Markdown notes and PDFs into a local vector store and retrieves relevant chunks by natural language query. Optimised for NVIDIA Jetson Orin Nano Super (JetPack 6.2); also runs on macOS/x86.

Retrieval reaches the vector store through a pluggable `RetrievalStore` seam (`src/rag/store/base.py`) rather than talking to ChromaDB directly — today the only implementation is `ChromaStore`.

## What it indexes

- **Obsidian vault** — Markdown notes with YAML frontmatter (type, domain, subdomain, status, source, confidence, tags, wikilinks)
- **PDF books** — technical books from your `Books/` directory
- **PDF resources** — papers, guides, and reference materials from your `Resources/` directory
- **Pre-extracted JSON** — `indexed/*.json` produced by this repo's own extraction pipeline (`src/extractor/`): full text + enriched metadata (title, topic→domain, tags, confidence). Deterministic, so re-runs are idempotent — this is the **primary** path for the book/resource library, and live PDF parsing is only a fallback for files the extractor hasn't processed yet

Indexing and retrieval are fully offline — no cloud APIs. The one exception is the **optional** `/answer` endpoint, which calls out to Anthropic or OpenAI to synthesise a grounded answer over retrieved chunks. It is off by default and needs both a `generation` config block and a provider API key; retrieval never calls out.

## Requirements

- Python 3.10+ (3.10 on Jetson, 3.12+ on macOS/x86)
- [uv](https://github.com/astral-sh/uv) — for environment management

## Setup

### macOS / x86

```bash
cd personal-rag
make install
# equivalent to:
#   uv venv .venv
#   uv pip install -r requirements.txt
#   uv pip install -e . --no-deps
```

### Jetson Orin Nano Super (JetPack 6.2, CUDA 12.6)

PyTorch must come from the Jetson AI Lab index — standard PyPI wheels are x86 only:

```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
pip install -r requirements-jetson.txt
pip install -e . --no-deps
```

## Configuration

Paths are set in `config.yaml`. To override them without editing the file, create a `.env` file (see `.env.example`) or set environment variables directly:

| Variable | Overrides |
|---|---|
| `RAG_VAULT_PATH` | `vault_path` in config.yaml |
| `RAG_PDF_BOOKS_PATH` | pdf_sources entry with `type: book` |
| `RAG_PDF_RESOURCES_PATH` | pdf_sources entry with `type: resource` |
| `RAG_JSON_PATH` | json_sources directory (pre-extracted document JSON) |
| `RAG_INDEX_PATH` | `index_path` (vector store dir) |

```bash
# .env example
RAG_VAULT_PATH=/Users/you/Documents/personal_knowledge/Career Knowledge Base/
RAG_PDF_BOOKS_PATH=/Users/you/Documents/personal_knowledge/Books/
RAG_PDF_RESOURCES_PATH=/Users/you/Documents/personal_knowledge/Resources/
RAG_JSON_PATH=/Users/you/Documents/knowledge-base-index/indexed
```

Point `RAG_VAULT_PATH` at a **local** vault checkout. Indexing a cloud-synced mirror (Google Drive, Dropbox) is not supported — a divergent copy will produce a divergent index.

`.env.example` documents a second, separate layer of `RAG_*` variables used only by Docker Compose to resolve host-side volume mounts for the extractor pipeline. Those are not read by the Python app.

All other settings (chunk size, embedding model, excluded dirs) have sensible defaults in `config.yaml`. Full reference: [docs/configuration.md](docs/configuration.md).

### Config profiles (multiple corpora)

`config.yaml` is one profile; `config.personal.yaml` (personal KB, light/Jetson)
and `config.logmanager.yaml` (Logmanager wiki, heavy/x86, markdown-only) are two
more. Select a profile per instance with `RAG_CONFIG_PATH` — no code change
needed:

```bash
RAG_CONFIG_PATH=./config.personal.yaml .venv/bin/rag-index
RAG_CONFIG_PATH=./config.logmanager.yaml .venv/bin/rag-serve
```

Full comparison and two-instance run details: [docs/configuration.md § Config profiles](docs/configuration.md#config-profiles).

## Usage

### Index your content

```bash
make index
# or: .venv/bin/rag-index
```

Indexing is fully streaming — each file is extracted, embedded, and upserted to ChromaDB before the next file starts. No global chunk accumulation in RAM. Uses GPU if CUDA is available, otherwise CPU.

Expect ~5–10 minutes for a large vault + book library on first run.

### Query

```bash
make query Q="What do I know about Kubernetes?"
# or: .venv/bin/rag-query "your question" -n 12
```

**Metadata filters** narrow results before ranking:

```bash
.venv/bin/rag-query "container orchestration" --domain DevOps
.venv/bin/rag-query "testing" --domain "Software Engineering" --subdomain "Python & Backend Development"
.venv/bin/rag-query "container orchestration" --domain DevOps --confidence high
.venv/bin/rag-query "neural networks" --source pdf --type book
.venv/bin/rag-query "deployment" --status processed
.venv/bin/rag-query "kubernetes" --tag devops --tag containers   # repeatable; multiple --tag = AND
.venv/bin/rag-query "RAG pipeline" --json
```

`--domain`, `--subdomain`, `--type`, `--source`, `--confidence` and `--status` are native store-level filters. `--tag` is a case-insensitive exact-match post-filter over a widened candidate pool (`tag_fetch_k`), because tags are stored as a joined string.

**Reranking** — a cross-encoder can reorder the dense candidates:

```bash
.venv/bin/rag-query "how do I rotate secrets" --rerank
.venv/bin/rag-query "how do I rotate secrets" --no-rerank
```

With neither flag, the profile's `rerank_default` decides. It is **off** in `config.yaml` and `config.personal.yaml` (measured: it loses overall recall@5 on the personal corpus) and **on** in `config.logmanager.yaml`. See CLAUDE.md roadmap item 5 for the measurements.

Pass `--help` to see all options.

**Output per result:**
- Title and section heading
- File path (note or PDF filename)
- Type, domain, subdomain, status metadata
- Semantic distance score (lower = more relevant)
- Chunk text preview

### Tests

Two separate suites — the offline unit suite is what the CI badge above measures:

```bash
make test-unit    # offline pytest suite — no network, no real model, no index needed
```

```bash
make test         # retrieval smoke test — needs a populated index
make test K=devops    # filter by keyword
```

`make test-unit` covers chunking, extractors, the incremental indexing engine, the store seam, config profiles, the `search()` seam, the eval harness, the generation layer, and the HTTP API. Run `make test` after every reindex to catch quality regressions; results below distance 0.75 are marked OK, above are marked WEAK.

**Note:** `pytest` is not in the lockfile, so a venv built by `make install` needs it added:

```bash
uv pip install pytest
```

### Measure retrieval quality

```bash
make eval    # recall@5 / recall@10 / MRR over tests/eval/golden_queries.jsonl
```

45 labelled queries (30 vault, 15 resource), scored per-corpus. Snapshots live in `tests/eval/`.

### Makefile shortcuts

A `Makefile` wraps all common commands so you don't have to type the full paths. `make help` lists every target.

```bash
make index
make query Q="What do I know about Kubernetes?"
make test-unit                        # offline unit suite
make test K=devops                    # retrieval smoke test
make eval                             # recall@k / MRR
make serve                            # HTTP backend on 0.0.0.0:8000
make pipeline-status                   # extractor pipeline pre-flight

make build          # Docker x86
make docker-index
make docker-query Q="secrets in Python"
make docker-serve

make build-jetson   # Docker Jetson (run on Jetson)
make jetson-index
make jetson-query Q="bioprocessing workflows"
make jetson-serve
```

The extractor pipeline has its own targets (`make extract`, `enrich`, `build-index`, `build-notes`, `build-sqlite`, `build-books-index`, `link-mocs`, `dup-detect`, `search`), each with `docker-` and `jetson-` variants. See CLAUDE.md for the order to run them in.

## Docker

The repo ships two Dockerfiles and matching Compose files.

### Prerequisites

1. Copy `.env.example` to `.env` and set **all four** indexer paths:

```bash
cp .env.example .env
# edit .env — set RAG_VAULT_PATH, RAG_PDF_BOOKS_PATH, RAG_PDF_RESOURCES_PATH, RAG_JSON_PATH
```

> **Do not skip `RAG_JSON_PATH`.** Compose resolves unset mounts to the host's `/tmp`, so a missing variable silently mounts an empty directory and the indexer sees zero files from that source. A missing `RAG_VAULT_PATH`/`RAG_JSON_PATH` pair is what caused a full index wipe on 2026-07-15. The indexer now refuses to prune when *every* source reports 0 files, but it cannot detect a *partially* broken mount — verify the startup log reports non-zero counts per source.

Running the extractor pipeline in Docker needs a further three host-side variables (`RAG_OUTPUT_PATH`, `RAG_CATALOG_PATH`, `RAG_MOCS_PATH`); see the comments in `.env.example`.

2. Source paths are mounted read-only into the container; the vector store and the HuggingFace model cache live in named Docker volumes that survive restarts.

> **Config profiles are not available in Docker.** Both images copy only `config.yaml`, and neither Compose file passes `RAG_CONFIG_PATH` — so `config.personal.yaml` / `config.logmanager.yaml` work on a host venv only.

### x86 / macOS

```bash
make build
make docker-index
make docker-query Q="What is K3s?"
make docker-test
```

Or directly:

```bash
docker compose run --rm rag python -m rag.indexer
docker compose run --rm rag python -m rag.query "your question" --domain DevOps
```

### Jetson Orin Nano Super (JetPack 6.2)

**Build and run on the Jetson itself.** PyTorch wheels are aarch64-only and will not install on x86.

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

```bash
make build-jetson
make jetson-index
make jetson-query Q="What do I know about bioprocessing?"
```

Or directly:

```bash
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query "your question"
```

The first `build-jetson` will be slow (~1.5 GB PyTorch layer). Subsequent builds reuse the cached layer.

## Backend API

An HTTP backend (`rag-serve`) serves semantic queries over the same index. It loads the embedding model, the store collection, the reranker and (if configured) the answer generator **once** at startup and keeps them resident, so the internal bots (Telegram / Wiki) don't pay the CLI's per-query cold-start. It reuses `query.search()` — retrieval is never reimplemented. It runs on the Jetson, is reached over Tailscale (internal only), and uses JWT bearer tokens for auth — there is no in-app TLS.

Set `RAG_API_JWT_SECRET` (a strong secret, ≥32 bytes) in `.env`, then start the server:

```bash
make serve            # local:   .venv/bin/rag-serve  (0.0.0.0:8000)
make docker-serve     # x86:     docker compose up api
make jetson-serve     # Jetson:  docker compose -f docker-compose.jetson.yml up api
```

Mint a service token and make an authenticated query:

```bash
export TOKEN=$(RAG_API_JWT_SECRET=<secret> rag-token --subject my-bot)

curl -s -X POST http://gpu-01:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does K3s handle secrets?", "n_results": 5, "filters": {"domain": "DevOps"}}'
```

`GET /health` is unauthenticated; `GET /status`, `POST /query`, `POST /answer`, `POST /index`, and `GET /index/jobs/{id}` require the token. See [docs/api.md](docs/api.md) for every endpoint, request/response schemas, the JWT flow, and a copy-paste Python client for the bots.

### Grounded answers (optional)

`POST /answer` runs the same retrieval, then asks an LLM to synthesise a cited answer over the retrieved chunks. This is the **only** part of the project that leaves the machine.

It is **disabled by default**. To enable it, add a `generation` block to your config profile and export the provider key:

```yaml
generation:
  provider: anthropic        # anthropic | openai
  model: claude-sonnet-5
  max_tokens: 1024
  temperature: 0.0
  api_key_env: ANTHROPIC_API_KEY   # the key is read from the env, never stored in config
```

With no `generation` block or no key exported, `/answer` returns 503 and `/query` is unaffected. The generator sits *above* `search()` and is orthogonal to the store backend.

## How it works

```
Obsidian vault (.md)  ─┐
PDF books & resources  ─┼─► Per-file text extraction (ThreadPoolExecutor, md_workers / pdf_workers)
Pre-extracted JSON     ─┘         │
                                   │  one file at a time
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
                        chunk IDs are content hashes, so unchanged chunks are
                        skipped and stale ones pruned — the collection is
                        never wiped
                        free memory → next file
                                   │
                    ┌──────────────┴───────────────┐
                    │  src/rag/query.py  search()  │  ← CLI, /query, /answer, eval
                    └──────────────┬───────────────┘
                                   ▼
                        Metadata filter (--domain / --subdomain / --type /
                        --source / --confidence / --status / --tag)
                                   │
                                   ▼
                        Top-K chunks by cosine similarity
                                   │
                                   ▼
                        Optional cross-encoder rerank → top-N
                                   │
                                   ▼
                        Optional LLM synthesis (/answer only, off by default)
```

## Metadata per chunk

| Field | Source | Example |
|---|---|---|
| `path` | file path | `Knowledge/DevOps/K3s.md` |
| `title` | frontmatter or PDF metadata | `K3s Adoption Decision Framework` |
| `heading` | Markdown heading or page range | `Summary` / `p.12-15` |
| `type` | frontmatter or pdf_source type | `Knowledge`, `book`, `resource` |
| `domain` | frontmatter or path | `DevOps` |
| `subdomain` | frontmatter or subfolder | `Python & Backend Development` |
| `status` | frontmatter | `processed` |
| `source` | frontmatter or `pdf` | `ChatGPT`, `pdf` |
| `confidence` | frontmatter (empty for PDFs) | `high`, `medium` |
| `tags` | frontmatter (comma-joined) | `kubernetes, containers` |
| `wikilinks` | extracted from body | `K3s, Docker` |

## Project files

| Path | Purpose |
|---|---|
| `src/rag/utils.py` | Shared helpers — `load_config()`, logging setup, telemetry suppression, env overrides |
| `src/rag/chunking.py` | Pure text helpers — heading split, char chunking, stable IDs, wikilinks |
| `src/rag/extractors/` | One module per source type (`markdown`, `pdf`, `json_doc`) + `iter_sources()` registry |
| `src/rag/indexing.py` | Incremental engine — embed/upsert, per-file diff, stale prune, `run_source()` |
| `src/rag/indexer.py` | `main()` orchestration — MD + PDF + JSON → store; CLI entry: `rag-index` |
| `src/rag/query.py` | `search()` seam + query CLI with metadata filters, rerank, JSON output; CLI entry: `rag-query` |
| `src/rag/store/` | Pluggable `RetrievalStore` Protocol (`base.py`) + `ChromaStore` — the only place `chromadb` is imported |
| `src/rag/generation/` | Optional answer synthesis above `search()` — Anthropic/OpenAI over httpx, `[n]` citations |
| `src/rag/api/` | FastAPI backend — lifespan model loading, JWT auth, `/query` `/answer` `/index`; entries: `rag-serve`, `rag-token` |
| `src/rag/eval.py` | recall@k / MRR harness over the golden query set |
| `src/rag/pipeline_status.py` | Pre-flight check of the extractor pipeline's inputs/outputs |
| `src/extractor/` | Document extraction pipeline — PDF/EPUB/MD → JSON, metadata enrichment, SQLite FTS, Obsidian note generation |
| `tests/` | Offline pytest suite (`make test-unit`) + `test_queries.py`, the index-dependent smoke script |
| `tests/eval/` | Golden query set and recorded eval snapshots |
| `scripts/` | `eval_recall.py` (eval shim), `drop_collections.py` (drop trial collections) |
| `Dockerfile` | x86 / macOS container image |
| `Dockerfile.jetson` | Jetson JetPack 6.2 container image (build on Jetson) |
| `docker-compose.yml` | x86 Compose file with volume mounts |
| `docker-compose.jetson.yml` | Jetson Compose file (`runtime: nvidia`) |
| `config.yaml` | Default profile — vault paths, PDF sources, chunk settings |
| `config.personal.yaml` | Personal-KB profile (light/Jetson) — select via `RAG_CONFIG_PATH` |
| `config.logmanager.yaml` | Logmanager wiki profile (heavy/x86, markdown-only) — select via `RAG_CONFIG_PATH` |
| `pyproject.toml` | Package definition and all 16 `rag-*` console entry points |
| `.env.example` | Template for path overrides via environment variables |
| `Makefile` | Shortcuts for local and Docker workflows |
| `requirements.txt` | Full pinned dependency lockfile (macOS/x86) |
| `requirements-direct.txt` | Direct dependencies only (use with `uv pip compile` to regenerate lockfile) |
| `requirements-jetson.txt` | Direct dependencies for Jetson JetPack 6.2 (aarch64, CUDA 12.6) |
| `docs/architecture.md` | Pipeline walkthrough and design notes |
| `docs/configuration.md` | Full config.yaml reference and env var overrides |
| `docs/jetson.md` | Jetson setup guide |

## Docs

| Document | Contents |
|---|---|
| [CLAUDE.md](CLAUDE.md) | **Start here** — current corpus state, indexing behaviour, gotchas, quality roadmap |
| [docs/api.md](docs/api.md) | Backend HTTP API — endpoints, JWT auth, service tokens, curl + Python client |
| [docs/architecture.md](docs/architecture.md) | Pipeline walkthrough, streaming design, chunk IDs, store seam, rerank, generation |
| [docs/configuration.md](docs/configuration.md) | Full config reference, env var overrides, hardware tuning, config profiles |
| [docs/jetson.md](docs/jetson.md) | Jetson install guide, Docker, memory budget, GPU constraints |
| [docs/ADR-multi-corpus-profiles-and-pluggable-store.md](docs/ADR-multi-corpus-profiles-and-pluggable-store.md) | Why config profiles and the `RetrievalStore` seam exist |
| [docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md](docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md) | Planned OpenSearch backend (not built — hybrid BM25 lands here) |
| [docs/archive/](docs/archive/) | Historical reports and completed plans — **not** current reference |

## Known issues

**ChromaDB telemetry warnings** — ChromaDB 0.6.3 and posthog 7.x have a signature mismatch. `utils.py` suppresses it at import time. Safe to ignore; do not remove the patch when upgrading chromadb until confirmed fixed.

**Encrypted PDFs** — AES-encrypted PDFs are decrypted transparently using a blank owner password (the common publish-lock pattern). This requires the `cryptography` package (`>=3.1`), which is included in both `requirements.txt` and `requirements-jetson.txt`. PDFs that require a non-blank password are skipped with a warning.

**Unresolved Obsidian template vars** — notes with `{{DATE}}` or similar unfilled placeholders are handled by stripping `{{...}}` before YAML parsing.

**Broken YAML frontmatter silently drops a note** — an unquoted colon in a frontmatter value makes the parse raise, and the note is logged as one `SKIP` line among thousands and indexed with zero chunks. It becomes unsearchable, not merely stale. Audit after a run:

```bash
grep -c "SKIP .*mapping values" logs/rag.log
```

**Reindex job records are never pruned** — the API keeps one record per reindex for the life of the server process. Harmless in practice (reindexes are rare), but a very long-lived server accumulates them.
