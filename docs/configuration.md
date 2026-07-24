# Configuration Reference

## config.yaml

All settings live in `config.yaml` at the project root. User-specific paths can be overridden via environment variables without editing this file — see [Environment variables](#environment-variables) below.

```yaml
vault_path: '/path/to/Obsidian Vault/'
index_path: "./chroma_db"
log_path: "./logs/rag.log"
collection_name: "obsidian_markdown"

exclude_dirs:
  - ".obsidian"
  - ".trash"
  - "Resources/_catalog"
  - "Resources/Generated"
  - "Attachments"
  - "Archive"
  - "Templates"
  - ".git"

exclude_files:
  - ".DS_Store"
  - "CLAUDE.md"
  - "Dashboard.md"
  - "Home.md"
  - "README.md"

exclude_filename_patterns:
  - "* MOC.md"

store: chroma

chunk_max_chars: 1200
chunk_overlap_chars: 150
embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
embedding_batch_size: 16
markdown_workers: 1
pdf_workers: 1
embedding_workers: 1   # legacy, reserved

pdf_sources:
  - path: '/path/to/Books/'
    type: book
  - path: '/path/to/Resources/'
    type: resource

json_sources:
  - path: '~/Documents/knowledge-base-index/indexed'
```

### Field reference

| Key | Default | Description |
|---|---|---|
| `vault_path` | — | Obsidian vault root. Supports `~` and spaces. |
| `index_path` | `./chroma_db` | ChromaDB persistent storage directory. |
| `log_path` | `./logs/rag.log` | App log file (rotating, 5 MB × 3). Console output is unaffected. |
| `log_db_path` | `<log_path>.sqlite` | Structured SQLite log DB (`logs` table). Defaults alongside `log_path`. |
| `store` | `chroma` | Retrieval backend selector. Only `chroma` exists today (`opensearch` is a planned future backend for the Logmanager wiki corpus — see [ADR-multi-corpus-profiles-and-pluggable-store.md](ADR-multi-corpus-profiles-and-pluggable-store.md)). Adding this key is behavior-preserving; omitting it also defaults to `chroma`. |
| `collection_name` | `obsidian_markdown` | ChromaDB collection name. |
| `exclude_dirs` | see above | Vault subdirectories to skip during indexing. `Resources/Generated` holds auto-generated catalog stubs (Resource Notes, Topic MOCs, Learning Paths) whose underlying book/resource content is already fully indexed via `json_sources`. |
| `exclude_files` | `.DS_Store`, `CLAUDE.md`, `Dashboard.md`, `Home.md`, `README.md` | Filenames to skip regardless of directory (e.g. vault hub/navigation pages). |
| `exclude_filename_patterns` | `* MOC.md` | fnmatch patterns applied to filenames in any directory. Used to skip domain MOCs — navigation-only wikilink indexes whose chunks would match many queries spuriously. The per-note link graph is still captured in the `wikilinks` metadata field. |
| `chunk_max_chars` | `1200` | Maximum characters per chunk. Smaller = lower per-file RAM peak. |
| `chunk_overlap_chars` | `150` | Character overlap between consecutive chunks. |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformers model name or HuggingFace path. |
| `embedding_batch_size` | `16` | Chunks per `model.encode()` call. Keep at 16 on Jetson (8 GB unified RAM). |
| `markdown_workers` | `1` | ThreadPoolExecutor threads for MD extraction. 1 = sequential. |
| `pdf_workers` | `1` | ThreadPoolExecutor threads for PDF extraction. Keep at 1 on Jetson. |
| `pdf_sources` | `[]` | List of `{path, type}` PDF source directories. `type` is stored as chunk metadata. Acts as a **fallback**: files whose name is already covered by a `json_sources` document are skipped, so configuring both never double-indexes. |
| `json_sources` | `[]` | List of `{path}` dirs of pre-extracted document JSON (`doc-text-extractor` `indexed/*.json`): full `text` + metadata (title, primary_topic→domain, resource_type→type, tags, confidence). Deterministic, so re-runs are idempotent. |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used at query time (no reindex). Default-on; `rag-query --no-rerank` disables. |
| `rerank_fetch_k` | `20` | Dense candidate pool retrieved before reranking down to `n_results`. |
| `tag_fetch_k` | `200` | Dense pool floor when a `--tag` filter is active (tags are post-filtered — see roadmap item 7). Only applies when tags are supplied. |
| `generation` | *(absent)* | Optional block enabling the `/answer` RAG endpoint. Absent ⇒ `/answer` returns `503`, `/query` unaffected. Sub-keys: `provider` (`anthropic`\|`openai`), `model` (required), `max_tokens` (`1024`), `temperature` (`0.0`), `timeout` (`60`), `api_key_env` (provider default: `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`), `base_url` (optional; OpenAI-compat servers). The API **key** is read from the env var named by `api_key_env`, never from this file. See [api.md](api.md#enabling-answer-generation). |

## Environment variables

Override any path without editing `config.yaml`. Copy `.env.example` to `.env` — it is loaded automatically at startup via `python-dotenv`.

| Variable | Overrides | Notes |
|---|---|---|
| `RAG_VAULT_PATH` | `vault_path` | |
| `RAG_PDF_BOOKS_PATH` | pdf_sources entry with `type: book` | |
| `RAG_PDF_RESOURCES_PATH` | pdf_sources entry with `type: resource` | |
| `RAG_JSON_PATH` | `json_sources` (single dir) | Set to `/index` automatically in Docker (bind-mount your `indexed/` dir). |
| `RAG_INDEX_PATH` | `index_path` | Set to `/data/chroma` automatically in Docker. |
| `RAG_LOG_PATH` | `log_path` | Set to `/data/logs/rag.log` automatically in Docker (bind-mounted to `./logs`). |
| `RAG_LOG_DB_PATH` | `log_db_path` | SQLite log DB path. Defaults to the text log path with a `.sqlite` suffix. |
| `RAG_CONFIG_PATH` | Path to `config.yaml` itself | Useful when running from a directory other than the project root. |
| `RAG_API_JWT_SECRET` | — (API only) | HS256 shared secret for the backend API (`rag-serve` / `rag-token`). **Required** to serve — protected routes return 500 if unset; no default. Use a strong secret (≥32 bytes; PyJWT warns on short keys). Only the API server needs it. |
| `RAG_API_HOST` | — (API only) | Bind address for the API server. Default `0.0.0.0`. |
| `RAG_API_PORT` | — (API only) | Bind port for the API server. Default `8000`. |

### .env example

```bash
RAG_VAULT_PATH=/Volumes/Drive/mindmap/Career Knowledge Base/
RAG_PDF_BOOKS_PATH=/Volumes/Drive/mindmap/Books/
RAG_PDF_RESOURCES_PATH=/Volumes/Drive/mindmap/Resources/
```

## Tuning for different hardware

| Hardware | `chunk_max_chars` | `embedding_batch_size` | `markdown_workers` | `pdf_workers` |
|---|---|---|---|---|
| Jetson Orin Nano Super (8 GB) | 1200 | 16 | 1 | 1 |
| Desktop (16+ GB RAM, no GPU) | 1800 | 32 | 4 | 2 |
| Desktop (NVIDIA GPU, 8+ GB VRAM) | 1800 | 64 | 4 | 2 |

## Config profiles

Full design context: [ADR-multi-corpus-profiles-and-pluggable-store.md](ADR-multi-corpus-profiles-and-pluggable-store.md).

`config.yaml` is one **profile** among several. A profile is just a config file
selected at deploy/run time via `RAG_CONFIG_PATH` (see the env var table above) —
`load_config()`'s `_find_config()` already resolves `RAG_CONFIG_PATH` with no
fallback-to-default when set, so pointing it at a different file requires **no
code change**. Two profiles ship today, run as two independent instances:

| Profile | File | Corpus | Hardware |
|---|---|---|---|
| **personal** (light) | `config.personal.yaml` | Personal KB: Obsidian vault + Books/Resources (PDF + pre-extracted JSON) | Jetson Orin Nano (8 GB) |
| **logmanager** (heavy) | `config.logmanager.yaml` | Logmanager wiki, markdown-only | x86 server, no RAM ceiling |
| *(default)* | `config.yaml` | Same corpus/content as `config.personal.yaml` today (historical default) | macOS/x86 dev, Jetson |

### Key knob differences

| Knob | `config.personal.yaml` | `config.logmanager.yaml` |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | `all-MiniLM-L6-v2` (**placeholder** — see file comment; a larger model is deferred until the wiki corpus exists and can be evaluated, to avoid repeating the bge-small/gte-small net-regression seen on the personal corpus) |
| `reranker_model` / `rerank_fetch_k` | ms-marco-MiniLM-L-6 / 20 | same model / 40 (tunable, or disable with `--no-rerank` if it hurts) |
| `tag_fetch_k` | 200 | 400 |
| `embedding_batch_size` | 16 | 64 |
| `*_workers` | 1 | `markdown_workers: 8`, `embedding_workers: 4` |
| `collection_name` | `obsidian_markdown` | `wiki_lm` |
| `index_path` | `./chroma_db` | `./chroma_db_wiki` |
| `vault_path` | personal vault checkout | wiki repo checkout (placeholder path — update once the wiki checkout exists) |
| `pdf_sources` / `json_sources` | both set (Books/Resources + pre-extracted JSON) | **absent** — wiki corpus is markdown-only |
| `extractor:` block | present (book/resource pipeline paths) | **omitted** — the extraction/enrich/build-index/Resource-Notes/MOC-backlink pipeline is personal-KB-specific and is not run for the wiki |
| `store` | `chroma` | `chroma` |

### Running two instances

Each instance is a normal `rag-index` / `rag-serve` process pointed at its own
profile via `RAG_CONFIG_PATH`. They never share `index_path` or
`collection_name`, so they can run concurrently on the same or different hosts
without interfering with each other:

```bash
# Personal KB instance (Jetson-sized)
RAG_CONFIG_PATH=./config.personal.yaml .venv/bin/rag-index
RAG_CONFIG_PATH=./config.personal.yaml .venv/bin/rag-serve

# Logmanager wiki instance (x86, markdown-only)
RAG_CONFIG_PATH=./config.logmanager.yaml .venv/bin/rag-index
RAG_CONFIG_PATH=./config.logmanager.yaml .venv/bin/rag-serve
```

`RAG_VAULT_PATH` / `RAG_INDEX_PATH` / `RAG_JSON_PATH` / etc. still apply as
per-instance overrides on top of whichever profile `RAG_CONFIG_PATH` selects —
useful for Docker Compose deployments where each instance's `.env` sets the
mount paths but shares the same profile file baked into the image.

### `store` key

`store: chroma` selects the retrieval backend. It exists in all three config
files today (`config.yaml`, `config.personal.yaml`, `config.logmanager.yaml`)
and defaults to `chroma` when absent — adding it changes no behavior. It is
forward-looking: the ADR's Axis 2 introduces a `RetrievalStore` abstraction so
an `opensearch` backend can be added later (OpenSearch is the likely production
target for the Logmanager wiki chatbot) without an engine rewrite. No other
backend exists yet.
