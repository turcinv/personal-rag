# personal-rag

[![tests](https://github.com/turcinvBenemeat/personal-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/turcinvBenemeat/personal-rag/actions/workflows/ci.yml)

Local semantic search over an Obsidian knowledge vault and PDF book library. Indexes Markdown notes and PDFs into a local ChromaDB vector store and retrieves relevant chunks by natural language query. Runs fully offline — optimised for NVIDIA Jetson Orin Nano Super (JetPack 6.2) and macOS/x86.

## What it indexes

- **Obsidian vault** — Markdown notes with YAML frontmatter (type, domain, status, source, confidence, tags, wikilinks)
- **PDF books** — technical books from `mindmap/Books/`
- **PDF resources** — papers, guides, and reference materials from `mindmap/Resources/`
- **Pre-extracted JSON** *(optional)* — `indexed/*.json` from the external `doc-text-extractor` pipeline (full text + enriched metadata: title, topic→domain, tags, confidence). Deterministic, so re-runs are idempotent — preferred over live PDF parsing for the book/resource library

All content is indexed locally. Nothing is sent to any cloud service.

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
| `RAG_INDEX_PATH` | `index_path` (ChromaDB storage dir) |

```bash
# .env example
RAG_VAULT_PATH=/Volumes/Drive/mindmap/Career Knowledge Base/
RAG_PDF_BOOKS_PATH=/Volumes/Drive/mindmap/Books/
RAG_PDF_RESOURCES_PATH=/Volumes/Drive/mindmap/Resources/
```

All other settings (chunk size, embedding model, excluded dirs) have sensible defaults in `config.yaml`.

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

**Metadata filters** narrow results to a specific domain, type, source, or confidence:

```bash
.venv/bin/rag-query "container orchestration" --domain DevOps
.venv/bin/rag-query "container orchestration" --domain DevOps --confidence high
.venv/bin/rag-query "neural networks" --source pdf --type book
.venv/bin/rag-query "RAG pipeline" --json
```

Pass `--help` to see all options.

**Output per result:**
- Title and section heading
- File path (note or PDF filename)
- Type, domain, status metadata
- Semantic distance score (lower = more relevant)
- Chunk text preview

### Smoke test retrieval quality

```bash
make test
make test K=devops    # filter by keyword
# or: .venv/bin/python tests/test_queries.py [keyword]
```

Run this after every reindex to catch quality regressions. Results below distance 0.75 are marked OK; above are marked WEAK.

### Makefile shortcuts

A `Makefile` wraps all common commands so you don't have to type the full paths:

```bash
make index
make query Q="What do I know about Kubernetes?"
make test K=devops

make build          # Docker x86
make docker-index
make docker-query Q="secrets in Python"

make build-jetson   # Docker Jetson (run on Jetson)
make jetson-index
make jetson-query Q="bioprocessing workflows"
```

## Docker

The repo ships two Dockerfiles and matching Compose files.

### Prerequisites

1. Copy `.env.example` to `.env` and set your three paths:

```bash
cp .env.example .env
# edit .env — set RAG_VAULT_PATH, RAG_PDF_BOOKS_PATH, RAG_PDF_RESOURCES_PATH
```

2. Source paths are mounted read-only into the container; ChromaDB and the HuggingFace model cache are stored in named Docker volumes that survive restarts.

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

An HTTP backend (`rag-serve`) serves semantic queries over the same index. It loads the embedding model, ChromaDB collection, and reranker **once** at startup and keeps them resident, so the internal bots (Telegram / Wiki) don't pay the CLI's per-query cold-start. It runs on the Jetson, is reached over Tailscale (internal only), and uses JWT bearer tokens for auth — there is no in-app TLS.

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

`GET /health` is unauthenticated; `GET /status`, `POST /query`, `POST /index`, and `GET /index/jobs/{id}` require the token. See [docs/api.md](docs/api.md) for every endpoint, request/response schemas, the JWT flow, and a copy-paste Python client for the bots.

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
                        Upsert batch → ChromaDB (local, ./chroma_db)
                        free memory → next file
                                   │
                          query_obsidian.py
                                   │
                                   ▼
                        Top-N chunks by cosine similarity
                        optional metadata filter (--domain / --type / --source / --confidence)
```

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

## Project files

| Path | Purpose |
|---|---|
| `src/rag/utils.py` | Shared helpers — `load_config()`, logging setup, telemetry suppression, env overrides |
| `src/rag/chunking.py` | Pure text helpers — heading split, char chunking, stable IDs, wikilinks |
| `src/rag/extractors/` | One module per source type (`markdown`, `pdf`, `json_doc`) + `iter_sources()` registry |
| `src/rag/indexing.py` | Incremental engine — embed/upsert, per-file diff, stale prune, `run_source()` |
| `src/rag/indexer.py` | `main()` orchestration — MD + PDF + JSON → ChromaDB; CLI entry: `rag-index` |
| `src/rag/query.py` | Semantic query CLI with metadata filters and JSON output; CLI entry: `rag-query` |
| `tests/test_queries.py` | 13-query retrieval smoke tests across all vault domains |
| `Dockerfile` | x86 / macOS container image |
| `Dockerfile.jetson` | Jetson JetPack 6.2 container image (build on Jetson) |
| `docker-compose.yml` | x86 Compose file with volume mounts |
| `docker-compose.jetson.yml` | Jetson Compose file (`runtime: nvidia`) |
| `config.yaml` | Default profile — vault paths, PDF sources, chunk settings |
| `config.personal.yaml` | Personal-KB profile (light/Jetson) — select via `RAG_CONFIG_PATH` |
| `config.logmanager.yaml` | Logmanager wiki profile (heavy/x86, markdown-only) — select via `RAG_CONFIG_PATH` |
| `pyproject.toml` | Package definition and `rag-index` / `rag-query` entry points |
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
| [docs/api.md](docs/api.md) | Backend HTTP API — endpoints, JWT auth, service tokens, curl + Python client |
| [docs/architecture.md](docs/architecture.md) | Pipeline walkthrough, streaming design, chunk IDs, metadata fields |
| [docs/configuration.md](docs/configuration.md) | Full `config.yaml` reference, env var overrides, hardware tuning table |
| [docs/jetson.md](docs/jetson.md) | Jetson install guide, Docker, memory budget, GPU constraints |

## Known issues

**ChromaDB telemetry warnings** — ChromaDB 0.6.3 and posthog 7.x have a signature mismatch. `utils.py` suppresses it at import time. Safe to ignore; do not remove the patch when upgrading chromadb until confirmed fixed.

**Encrypted PDFs** — AES-encrypted PDFs are decrypted transparently using a blank owner password (the common publish-lock pattern). This requires the `cryptography` package (`>=3.1`), which is included in both `requirements.txt` and `requirements-jetson.txt`. PDFs that require a non-blank password are skipped with a warning.

**Unresolved Obsidian template vars** — notes with `{{DATE}}` or similar unfilled placeholders are handled by stripping `{{...}}` before YAML parsing.
