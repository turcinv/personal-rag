# Configuration Reference

### Typed loading, overlays, and preflight

Configuration is validated before use. Unknown keys, invalid worker/batch sizes,
invalid overlap settings, and malformed nested blocks fail with an actionable
error. Relative paths are resolved against the selected config file, not the
process working directory.

A profile may inherit another profile with a relative `extends` path:

```yaml
extends: config.yaml
rerank_default: false
```

`config.personal.yaml` uses this mechanism so the personal profile no longer
duplicates the complete historical `config.yaml`. `config.example.yaml` is the
portable starter profile for new machines.

Validate the active profile and inspect every resolved source without indexing:

```bash
make config-doctor
.venv/bin/rag-config --strict
.venv/bin/rag-config --json
```

Preflight states are `available`, `missing`, `unreadable`, and `disabled`.
`--strict` exits non-zero for missing or unreadable enabled sources.

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
  - ".claude"
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
query_instruction: ""
embedding_batch_size: 16
markdown_workers: 1
pdf_workers: 1

pdf_sources:
  - path: '/path/to/Books/'
    type: book
  - path: '/path/to/Resources/'
    type: resource

json_sources:
  - path: '~/Documents/knowledge-base-index/indexed'

extractor:
  books_path: '/path/to/Books/'
  resources_path: '/path/to/Resources/'
  catalog_path: '/path/to/Career Knowledge Base/Resources/_catalog'
  output_path: '~/Documents/knowledge-base-index'
  obsidian_notes_path: '/path/to/Career Knowledge Base/Resources/Generated/Resource Notes'
  mocs_path: '/path/to/Career Knowledge Base/Resources/Generated'
```

### Field reference

| Key | Default | Description |
|---|---|---|
| `vault_path` | — | Obsidian vault root. Supports `~` and spaces. |
| `index_path` | `./chroma_db` | ChromaDB persistent storage directory. |
| `log_path` | `./logs/rag.log` | App log file (rotating, 5 MB × 3). Console output is unaffected. |
| `log_db_path` | `<log_path>.sqlite` | Structured SQLite log DB (`logs` table). Defaults alongside `log_path`. |
| `log_retention_days` | `30` | Rows older than this are pruned from the SQLite log on startup (`0` disables pruning). Keeps the structured log bounded on a long-lived server. |
| `log_queries` | `false` | When false (default), the raw query text is **not** written to the logs — only its length. Set `true` to log verbatim queries for debugging. Privacy default for a personal KB. |
| `offline` | `false` | When true (or `RAG_OFFLINE` set), exports `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` before any model load so models load only from the local cache and a cold cache fails fast instead of reaching the network. Pair with `embedding_revision` to pin an exact snapshot. |
| `store` | `chroma` | Retrieval backend selector. Only `chroma` exists today (`opensearch` is a planned future backend for the Logmanager wiki corpus — see [ADR-multi-corpus-profiles-and-pluggable-store.md](ADR-multi-corpus-profiles-and-pluggable-store.md)). Adding this key is behavior-preserving; omitting it also defaults to `chroma`. |
| `collection_name` | `obsidian_markdown` | ChromaDB collection name. |
| `exclude_dirs` | see above | Vault subdirectories to skip during indexing. `Resources/Generated` holds auto-generated catalog stubs (Resource Notes, Topic MOCs, Learning Paths) whose underlying book/resource content is already fully indexed via `json_sources`. |
| `exclude_files` | `.DS_Store`, `CLAUDE.md`, `Dashboard.md`, `Home.md`, `README.md` | Filenames to skip regardless of directory (e.g. vault hub/navigation pages). |
| `exclude_filename_patterns` | `* MOC.md` | fnmatch patterns applied to filenames in any directory. Used to skip domain MOCs — navigation-only wikilink indexes whose chunks would match many queries spuriously. The per-note link graph is still captured in the `wikilinks` metadata field. |
| `chunk_max_chars` | `1200` | Maximum characters per chunk. Smaller = lower per-file RAM peak. |
| `chunk_overlap_chars` | `150` | Character overlap between consecutive chunks. |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformers model name or HuggingFace path. **Changing this requires a new `collection_name`** — chunk IDs hash the chunk text, not the vector, so the incremental engine would skip re-embedding and silently keep the old model's vectors. See [architecture.md](architecture.md#changing-the-embedding-model--new-collection-name-required). |
| `query_instruction` | `""` | Instruction prefix prepended to the **query only** (never to indexed passages) before embedding. Some retrieval models need one — bge-small wants `"Represent this sentence for searching relevant passages: "`. Empty for all-MiniLM and gte-small. |
| `embedding_batch_size` | `16` | Chunks per `model.encode()` call. Keep at 16 on Jetson (8 GB unified RAM). |
| `markdown_workers` | `1` | ThreadPoolExecutor threads for MD extraction. 1 = sequential. |
| `pdf_workers` | `1` | ThreadPoolExecutor threads for PDF extraction. Keep at 1 on Jetson. |
| `pdf_sources` | `[]` | List of `{path, type}` PDF source directories. `type` is stored as chunk metadata. Acts as a **fallback**: files whose name is already covered by a `json_sources` document are skipped, so configuring both never double-indexes. |
| `json_sources` | `[]` | List of `{path}` dirs of pre-extracted document JSON (`indexed/*.json`, produced by `src/extractor/`): full `text` + metadata (title, primary_topic→domain, resource_type→type, tags, confidence). Deterministic, so re-runs are idempotent. |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used at query time (no reindex). |
| `rerank_fetch_k` | `20` | Dense candidate pool retrieved before reranking down to `n_results`. |
| `rerank_default` | `false` (this profile) | Whether `rag-query` / `make eval` / `POST /query` / `POST /answer` rerank when the caller says nothing. **Per-profile, because the cross-encoder is corpus-dependent** — measured 2026-07-27 it *loses* overall recall@5 on the personal corpus (0.911 → 0.844), so it is off here and on in `config.logmanager.yaml`. Override per call: `rag-query --rerank` / `--no-rerank`, or `"rerank": true\|false` in the request body. Absent from a config ⇒ falls back to `true` (pre-2026-07-27 behaviour). See CLAUDE.md roadmap item 5. |
| `tag_fetch_k` | `200` | Dense pool floor when a `--tag` filter is active (tags are post-filtered — see roadmap item 7). Only applies when tags are supplied. |
| `generation` | *(absent)* | Optional block enabling the `/answer` RAG endpoint. Absent ⇒ `/answer` returns `503`, `/query` unaffected. Sub-keys: `provider` (`anthropic`\|`openai`), `model` (required), `max_tokens` (`1024`), `temperature` (`0.0`), `timeout` (`60`), `api_key_env` (provider default: `ANTHROPIC_API_KEY`/`OPENAI_API_KEY`), `base_url` (optional; OpenAI-compat servers). The API **key** is read from the env var named by `api_key_env`, never from this file. A `base_url` is validated as egress: remote hosts must be `https`, plain `http` is allowed only for a local inference server (localhost). An *invalid* generation block (unknown provider, insecure `base_url`, missing model) fails the server at startup; a *disabled* one (no block / no key) just makes `/answer` return 503. See [api.md](api.md#enabling-answer-generation). |
| `api` | *(defaults)* | Optional block for the HTTP backend. Sub-keys: `host` (`0.0.0.0`), `port` (`8000`), `inference_concurrency` (`1`) — bounds concurrent embed/rerank work so the Jetson's shared 8 GB is not exhausted; a request that can't get a slot gets `503`. `jwt_audience` / `jwt_issuer` (optional) — when set, the API enforces the `aud`/`iss` claims on every token (also settable via `RAG_API_JWT_AUDIENCE`/`RAG_API_JWT_ISSUER`). |

**API authorization scopes.** Tokens may carry a `scopes` claim (`query`,
`answer`, `index`) to restrict a credential to specific routes — mint one with
`rag-token --scope query`. A token with **no** `scopes` claim keeps full access,
so existing service tokens are unaffected. The reranker is only preloaded at
startup when `rerank_default` is true (the personal profile skips it); a request
that forces `rerank=true` loads it lazily. `rag-token`'s default lifetime is now
30 days (was 3650) — pass `--expires-days` for longer-lived service tokens.

### Pipeline coordinator

`rag-pipeline` is the single orchestration layer for extractor stages. It loads
the selected typed profile, resolves paths once, models stage dependencies and
outputs, and invokes the existing extractor modules with fixed argument arrays.
The individual `rag-extract` / `rag-enrich` / `rag-build-*` commands remain
available for compatibility.

```bash
rag-pipeline --dry-run                 # full plan and current output status
rag-pipeline --stage build-sqlite      # selected stage plus dependencies
rag-pipeline --stage enrich --no-deps  # only the requested stage
rag-pipeline --list-stages
```

Local, Docker, and Jetson Make targets all delegate to this coordinator, so
`RAG_CONFIG_PATH` and environment path overrides are honored consistently.

### The `extractor:` block

Paths used by the extraction pipeline (`src/extractor/`, the `rag-extract` /
`rag-enrich` / `rag-build-*` entry points). Unrelated to indexing and retrieval —
`vault_path` / `pdf_sources` / `json_sources` above are what the indexer reads.
The whole block may be omitted (a markdown-only profile like
`config.logmanager.yaml` does exactly that); every key falls back to empty.

| Key | Description | Env override |
|---|---|---|
| `extractor.books_path` | Source dir of book PDFs/EPUBs to extract. Must be **local disk**, never a cloud-synced folder. | `RAG_BOOKS_PATH` |
| `extractor.resources_path` | Source dir of resource PDFs/EPUBs. Same constraint. | `RAG_RESOURCES_PATH` |
| `extractor.catalog_path` | Dir holding `resource_inventory.jsonl` — the classification records. A file with no catalog row is silently excluded from `build-index`. | `RAG_CATALOG_PATH` |
| `extractor.output_path` | Where `text_output/`, `indexed/` and `resources.db` are written. This is the dir you rsync to the Jetson. | `RAG_OUTPUT_PATH` |
| `extractor.obsidian_notes_path` | Destination for generated Resource Notes. | `RAG_OBSIDIAN_NOTES_PATH` |
| `extractor.mocs_path` | Dir of Topic MOCs that `link-mocs` injects backlinks into. | `RAG_MOCS_PATH` |

Both the PDF glob and the extractor's listing are **non-recursive**, which is what
makes `Books/_superseded/` a working place to park a redundant pdf/epub twin.

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
| `RAG_API_JWT_AUDIENCE` | `api.jwt_audience` | When set, the API enforces the token `aud` claim. |
| `RAG_API_JWT_ISSUER` | `api.jwt_issuer` | When set, the API enforces the token `iss` claim. |
| `RAG_OFFLINE` | `offline` | When set, forces offline model loading (`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`). |
| `RAG_BOOKS_PATH` | `extractor.books_path` | Extractor pipeline only. |
| `RAG_RESOURCES_PATH` | `extractor.resources_path` | Extractor pipeline only. |
| `RAG_CATALOG_PATH` | `extractor.catalog_path` | Extractor pipeline only. |
| `RAG_OUTPUT_PATH` | `extractor.output_path` | Extractor pipeline only. |
| `RAG_OBSIDIAN_NOTES_PATH` | `extractor.obsidian_notes_path` | Extractor pipeline only. |
| `RAG_MOCS_PATH` | `extractor.mocs_path` | Extractor pipeline only. |

> **The extractor `RAG_*` variables do double duty**, and this is the one place it
> bites. The Python app reads them via `load_config()`, *and* Docker Compose reads
> them at the host shell level to resolve `${VAR:-/tmp}` in its `volumes:` section.
> An unset variable therefore does not raise — Compose mounts the host's `/tmp`, the
> container sees an empty directory, and every pipeline step reports MISSING while
> the data sits untouched on disk. Set all six before using any `docker-*` /
> `jetson-*` extractor target. (`RAG_OBSIDIAN_NOTES_PATH` is the exception: both
> Compose files hardcode it to `/mocs/Resource Notes` rather than interpolating it.)

### .env example

```bash
RAG_VAULT_PATH=/Users/you/Documents/personal_knowledge/Career Knowledge Base/
RAG_PDF_BOOKS_PATH=/Users/you/Documents/personal_knowledge/Books/
RAG_PDF_RESOURCES_PATH=/Users/you/Documents/personal_knowledge/Resources/
RAG_JSON_PATH=/Users/you/Documents/knowledge-base-index/indexed
```

**Set `RAG_JSON_PATH` too.** A missing `RAG_VAULT_PATH`/`RAG_JSON_PATH` pair caused
the 2026-07-15 index wipe on the Jetson (see CLAUDE.md). Reconciliation is now
source-scoped: missing, unreadable, degraded, disabled, and unexpectedly empty
sources preserve their owned chunks while healthy sources reconcile independently.
The all-sources-zero guard is an additional backstop. Verify the source census and
do not use prune override flags to bypass an unexplained mount problem.

Point these at **local disk**. Indexing a cloud-synced mirror (Google Drive,
Dropbox) is unsupported — the vault copy there has a divergent history, and these
paths drifted to exactly such a mirror once already.

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
| `reranker_model` / `rerank_fetch_k` | ms-marco-MiniLM-L-6 / 20 | same model / 40 |
| `rerank_default` | `false` — measured: rerank costs overall recall@5 (0.911 → 0.844) on this corpus | `true` — kept on until there is a wiki eval set; the personal-corpus finding is corpus-specific and may not transfer |
| `tag_fetch_k` | 200 | 400 |
| `embedding_batch_size` | 16 | 64 |
| `*_workers` | 1 | `markdown_workers: 8` |
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
