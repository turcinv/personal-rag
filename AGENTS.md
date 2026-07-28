# AGENTS.md

## Project overview

`personal-rag` is a local RAG (retrieval-augmented generation) pipeline: it embeds
an Obsidian vault + a PDF book/resource library into a local vector store and serves
semantic queries over them, by CLI and over HTTP. Indexing and retrieval are fully
offline. Runs on macOS/x86 for development and on an NVIDIA Jetson Orin Nano Super
(8 GB unified CPU+GPU RAM) for production embedding/serving — the Jetson's memory
budget is a real constraint on any change here, not a formality.

Five subsystems live in this repo:

| Path | Role |
|---|---|
| `src/rag/` | The indexer + `query.search()`, the single retrieval seam |
| `src/rag/store/` | `RetrievalStore` Protocol + `ChromaStore` — the **only** place `chromadb` is imported |
| `src/rag/api/` | FastAPI backend (`rag-serve`) — JWT auth, `/query`, `/answer`, `/index` |
| `src/rag/generation/` | Optional LLM answer synthesis *above* `search()`; off by default |
| `src/extractor/` | Document extraction pipeline: PDF/EPUB/MD → text → enriched catalog → Obsidian notes → SQLite FTS |

Two invariants worth internalising before changing anything:

- **Never reimplement retrieval.** The CLI, `/query`, `/answer` and the eval harness
  all go through `query.search()`. Add a caller, not a second path.
- **Never import a store client outside `src/rag/store/`.** A unit test in
  `tests/test_store.py` enforces this.

## Setup commands

Always use the local `.venv` — never bare `python`/`python3` (Conda base gets
picked up otherwise) and never system Python.

```bash
make install                       # uv venv .venv && uv pip install -r requirements.txt && uv pip install -e . --no-deps
uv pip install pytest              # make install does NOT install it — see Testing below
.venv/bin/rag-index                # reindex vault + PDFs into the store
.venv/bin/rag-query "your question"
```

Recreate a broken venv from scratch (interpreter symlinks can dangle after a
system Python upgrade/removal):

```bash
rm -rf .venv && uv venv .venv && uv pip install -r requirements.txt && uv pip install -e . --no-deps
```

## Key commands

```bash
make index                # rag-index — reindex vault + PDFs into the store
make query Q="..."        # semantic query
make eval                 # recall@5/@10 + MRR over tests/eval/golden_queries.jsonl
make serve                # rag-serve — HTTP backend on 0.0.0.0:8000 (needs RAG_API_JWT_SECRET)
make pipeline-status      # check all extraction pipeline outputs
make extract enrich build-index build-notes build-sqlite build-vault-index  # full extractor pipeline, in order
make sync-to-jetson JETSON_HOST=turcinv@<host>   # rsync extraction outputs to the Jetson
make help                 # every target
```

`rag-serve` and `rag-token` (mint a service JWT) are the two API entry points; there
are 16 `rag-*` console scripts in total, see `pyproject.toml`.

Jetson-side (run ON the Jetson, not from macOS — aarch64 PyTorch wheels won't build
on x86): `make build-jetson`, `make jetson-index`, `make jetson-query Q="..."`,
`make jetson-serve`.

Requires a `.env` in this directory with `RAG_VAULT_PATH` and `RAG_JSON_PATH` set to
real paths on the Jetson — if those are unset, `docker-compose.jetson.yml` silently
falls back to mounting `/tmp` (empty) and every source reports 0 files.

**The total-failure case is now guarded in code:** `indexer.main()` raises
`RuntimeError` and prunes nothing if every source reports 0 files while the index
already holds chunks (added after the 2026-07-15 wipe, in which 172,557 chunks were
pruned this way). You no longer need to race it manually.

The guard does **not** cover a *partially* broken mount — one source missing while
others are fine is indistinguishable from a legitimate deletion and will prune. So
still check the startup log's per-source file counts. The guard also cannot fire on
an already-empty index, so a fresh profile pointed at a bad path "succeeds" silently.

## Testing instructions

```bash
make test-unit             # offline pytest suite — no index, no data, no network, no real model
make test [K=keyword]      # retrieval smoke tests against a populated index (tests/test_queries.py)
```

`make test-unit` collects nine files and ~163 tests covering chunking, both
extractor packages, the incremental indexing engine, the store seam + config
profiles, the `search()` seam, the eval harness, the generation layer, and the HTTP
API. It is fully offline: the API tests inject a fake state via
`dependency_overrides`, generation uses an `httpx.MockTransport`, and the store tests
run a real `ChromaStore` over a temp dir. `tests/test_queries.py` is excluded from
collection (`tests/conftest.py`) because it needs a populated index and a real model
download.

**`pytest` is not in the lockfile** — `make install` uses `--no-deps`, so the `dev`
extra is skipped and a fresh venv cannot run the suite. Install it separately
(`uv pip install pytest`).

CI (`.github/workflows/ci.yml`) runs `pytest tests/ -q` on Python 3.10 (matches
Jetson JetPack 6.2) on every push/PR. Note CI does **not** test 3.12, which is what
macOS development typically runs, and there is no lint or type-check step.

Run the unit suite before committing any change under `src/` — especially
`chunking.py`, `indexing.py`, `store/`, or the extractors, which have the most test
coverage and the most Jetson-memory sensitivity.

## Code style

No enforced formatter/linter (no black/ruff config in this repo) — match the
existing style in the file you're editing. Python 3.10 syntax only (Jetson
JetPack 6.2 ships 3.10, not 3.12 — do not use 3.11+-only syntax).

## Security & gotchas

- **No cloud calls for indexing or retrieval** — audited clean as of 2026-07 (no
  `google.cloud`/`gsutil`/`boto3`/GCS SDK anywhere; the only `gcs_path` references are
  an inert legacy id-namespace string, not a live location). Keep it that way — don't
  reintroduce a cloud dependency in the indexing or retrieval path without discussing
  it first.
- **There is exactly one intentional outbound path:** `src/rag/generation/` calls
  `api.anthropic.com` or the OpenAI API over `httpx` to synthesise `/answer`
  responses. It is **off by default** (no `generation` config block and no API key ⇒
  `/answer` returns 503, `/query` unaffected), the key is read from the environment
  and never stored in config, and retrieval never invokes it. Any *new* egress needs
  discussing; note the old GCS-focused audit greps would not catch one.
- `config.yaml`'s `extractor.books_path`/`resources_path`/`pdf_sources` must point
  at real local disk (`~/Documents/personal_knowledge/Books/`, `.../Resources/`) —
  these drifted to a stale Google Drive mirror path once already (fixed 2026-07-13,
  commit `40f7c7d`). Don't reintroduce a cloud-synced-folder path here.
- Jetson memory is unified (CPU+GPU share 8 GB) — keep `embedding_batch_size`,
  `markdown_workers`, `pdf_workers` small (see `docs/jetson.md`); don't use
  `encode_multi_process` on Jetson (NvSCI IPC, not CUDA IPC — it fails).
- See `CLAUDE.md` → "Known Limitations & Improvement Roadmap" before assuming the
  current retrieval design is the intended end state. As of 2026-07-28 items 1, 3, 4,
  5 and 7 are **done** (triple-collection scheme removed; eval set built;
  heading-aware chunking; cross-encoder rerank; tag/status filters), item 2 is
  resolved as *rejected* (two stronger embedding models both net-regressed on this
  corpus — MiniLM kept), and **item 6 (BM25/lexical hybrid) is the one open gap** —
  specced in `docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md`, where it comes free with
  the backend.
- **Reranking is per-profile, and off on the personal corpus.** `search()` always
  defaults to `rerank=False`; callers resolve the profile default through
  `query.rerank_default(config)` (absent key ⇒ `True`). Measured 2026-07-27, rerank
  *loses* overall recall@5 here (0.911 → 0.844) by ejecting correct vault hits out of
  the top 5, so `config.yaml`/`config.personal.yaml` set `false` while
  `config.logmanager.yaml` keeps `true`. Don't "fix" this without re-running
  `make eval`.

## PR / commit conventions

No enforced format — short imperative-mood commit messages matching existing
history (e.g. "Point books_path/resources_path at local disk", "Remove dead GCS
call paths"). Run `make test-unit` before committing changes under `src/`.
