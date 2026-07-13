# AGENTS.md

## Project overview

`personal-rag` is a local RAG (retrieval-augmented generation) pipeline: it embeds
an Obsidian vault + a PDF book/resource library into ChromaDB and serves semantic
queries over them. Fully offline, no cloud APIs for indexing or retrieval. Runs on
macOS/x86 for development and on an NVIDIA Jetson Orin Nano Super (8 GB unified
CPU+GPU RAM) for production embedding/serving — the Jetson's memory budget is a
real constraint on any change here, not a formality.

Two halves live in this repo: `src/rag/` (the RAG indexer/query engine) and
`src/extractor/` (the document extraction pipeline: PDF/EPUB/MD -> text ->
enriched catalog -> Obsidian notes -> SQLite FTS index).

## Setup commands

Always use the local `.venv` — never bare `python`/`python3` (Conda base gets
picked up otherwise) and never system Python.

```bash
make install                       # uv venv .venv && uv pip install -r requirements.txt && uv pip install -e . --no-deps
.venv/bin/rag-index                # reindex vault + PDFs into ChromaDB
.venv/bin/rag-query "your question"
```

Recreate a broken venv from scratch (interpreter symlinks can dangle after a
system Python upgrade/removal):

```bash
rm -rf .venv && uv venv .venv && uv pip install -r requirements.txt && uv pip install -e . --no-deps
```

## Key commands

```bash
make index                # rag-index — reindex vault + PDFs into ChromaDB
make query Q="..."        # semantic query
make pipeline-status      # check all extraction pipeline outputs
make extract enrich build-index build-notes build-sqlite build-vault-index  # full extractor pipeline, in order
make sync-to-jetson JETSON_HOST=turcinv@<host>   # rsync extraction outputs to the Jetson
```

Jetson-side (run ON the Jetson, not from macOS — aarch64 PyTorch wheels won't build
on x86): `make build-jetson`, `make jetson-index`, `make jetson-query Q="..."`.
Requires a `.env` in this directory with `RAG_VAULT_PATH` and `RAG_JSON_PATH` set to
real paths on the Jetson — if those are unset, `docker-compose.jetson.yml` silently
falls back to mounting `/tmp` (empty), and the indexer will report 0 files from
every source without erroring. Always check the indexer's startup log reports
non-zero file counts before letting a run finish.

## Testing instructions

```bash
make test-unit             # offline pytest unit suite (chunking, extractors, indexing) — no index/data needed
make test [K=keyword]      # retrieval smoke tests against a populated index (tests/test_queries.py)
```

CI (`.github/workflows/ci.yml`) runs `make test-unit`-equivalent (`pytest tests/ -q`)
on Python 3.10 (matches Jetson JetPack 6.2) on every push/PR. Run the unit suite
before committing any change to `src/rag/chunking.py`, `src/rag/indexing.py`, or the
extractors — these have the most test coverage and the most Jetson-memory
sensitivity.

## Code style

No enforced formatter/linter (no black/ruff config in this repo) — match the
existing style in the file you're editing. Python 3.10 syntax only (Jetson
JetPack 6.2 ships 3.10, not 3.12 — do not use 3.11+-only syntax).

## Security & gotchas

- **No cloud calls of any kind for indexing/retrieval** — audited clean as of
  2026-07 (no `google.cloud`/`gsutil`/`boto3`/GCS SDK anywhere; the only `gcs_path`
  references are an inert legacy id-namespace string, not a live location). Keep it
  that way — don't reintroduce a cloud dependency without discussing it first.
- `config.yaml`'s `extractor.books_path`/`resources_path`/`pdf_sources` must point
  at real local disk (`~/Documents/personal_knowledge/Books/`, `.../Resources/`) —
  these drifted to a stale Google Drive mirror path once already (fixed 2026-07-13,
  commit `40f7c7d`). Don't reintroduce a cloud-synced-folder path here.
- Jetson memory is unified (CPU+GPU share 8 GB) — keep `embedding_batch_size`,
  `markdown_workers`, `pdf_workers` small (see `docs/jetson.md`); don't use
  `encode_multi_process` on Jetson (NvSCI IPC, not CUDA IPC — it fails).
- See `CLAUDE.md` → "Known Limitations & Improvement Roadmap" for known retrieval-
  quality gaps (dated embedding model, no reranking/hybrid search) before assuming
  the current retrieval design is the intended end state. (The wasteful triple-
  collection indexing is now removed — item 1 is done.)

## PR / commit conventions

No enforced format — short imperative-mood commit messages matching existing
history (e.g. "Point books_path/resources_path at local disk", "Remove dead GCS
call paths"). Run `make test-unit` before committing changes under `src/`.
