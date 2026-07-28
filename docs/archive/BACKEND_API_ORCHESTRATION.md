# Backend API for personal-rag — Orchestration Plan

## Context

`personal-rag` currently exposes retrieval only through the `rag-query` CLI, which
cold-starts the embedding model, Chroma collection, and cross-encoder reranker on
**every** invocation. Two network clients — the Telegram RAG Bot and the Logmanager
Wiki RAG Chatbot — need to call retrieval (and trigger reindexing) over HTTP. The
goal is a FastAPI backend that loads model + collection + reranker **once** at process
startup and reuses `query.py`'s existing seams (`search`, `build_where`, `get_model`,
`open_collection`, `get_reranker`) so nothing about retrieval is reimplemented. The
CLI stays working unchanged; the API is an additional entry point.

Spec: `docs/BACKEND_API_PLAN.md` (in repo). Work is executed by four subagents in
`.claude/agents/` (api-builder, test-author, docs-scribe, code-reviewer), orchestrated
in phases, committing after each phase, advancing only when the offline suite is green.

### Key findings from exploration (trust the code where it disagrees with the spec)

- **`query.search()`** (src/rag/query.py:81) already accepts `model=`, `collection=`,
  `config=`, `rerank=`, `filters=` and returns records `{document, metadata, distance,
  rank(, rerank_score)}`. `build_where(domain, type_, source, confidence, subdomain)`
  builds the Chroma where-dict. `get_model` / `get_reranker` are process-level caches;
  `open_collection(config, name=None)` opens the persistent collection. **Reuse all of
  these — do not reimplement.**
- **Indexer 0-files anti-wipe guard** lives in `indexer.main()` (src/rag/indexer.py:98,
  `RuntimeError` raised at the `if`-check on ~line 97): raises `RuntimeError` (nonzero
  exit) rather than pruning when every source reports 0 files but the collection is
  non-empty. Running the indexer as a **subprocess** (`python -m rag.indexer`) inherits
  this guard for free — a nonzero return code must surface as job `failed`.
- **Deps:** `fastapi==0.136.3`, `uvicorn==0.49.0`, `starlette==1.2.1` are ALREADY in
  `requirements.txt` (pulled transitively via chromadb). They are **absent** from
  `requirements-jetson.txt` (a hand-maintained minimal pin list, NOT uv-compiled).
  `pyjwt` is in **neither**. So: add `fastapi`, `uvicorn[standard]`, `pyjwt` to
  `requirements-direct.txt` and recompile `requirements.txt` with `uv pip compile`
  (uv is installed — use the version on the build host); hand-add the same three
  (pinned) to `requirements-jetson.txt`.
- **`/status` cannot reuse `pipeline_status.py`** — that module reports *extraction-
  pipeline file states* (STALE/MISSING per build step), not Chroma collection state.
  `/status` should report `collection.count()`, collection name, embedding-model name,
  reranker-model name directly from app state. (Plan-vs-code discrepancy; noted.)
- **Config/env:** `load_config()` reads `config.yaml` + `RAG_*` env overrides. New env
  vars `RAG_API_JWT_SECRET` (required, no default — 500/refuse if unset), `RAG_API_HOST`
  (default `0.0.0.0`), `RAG_API_PORT` (default `8000`) read via `os.environ` in `run()`.
- **Dockerfile CMD** = `["python","-m","rag.query","--help"]` in both Dockerfiles —
  MUST NOT change. Compose selects the server via a `command:` override.
- Test pattern: `tests/test_indexing.py` (FakeModel, temp PersistentClient) and
  `tests/test_query_search.py` (FakeColl, FakeReranker, monkeypatched `get_reranker`).
  Offline suite runs via `make test-unit` (`.venv/bin/python -m pytest tests/ -q`).
  `tests/conftest.py` already ignores the online `test_queries.py`
  (`collect_ignore = ["test_queries.py"]`).

### Working-tree handling (decided with user)

The branch base is dirty. Handle as:

1. **First commit on the branch = the unrelated `build-books-index` WIP** (pyproject.toml
   +1, Makefile +23/−7, .env.example +16, untracked `src/extractor/build_books_index.py`),
   committed on its own with a clear message so all later API commits stay clean.
2. **Ignore the mode-bit noise** (~30 files 100644→100755, zero content) — never
   `git add -A`; always stage explicit paths.
3. **Commit the task artifacts** `.claude/` (four agent defs) + `docs/BACKEND_API_PLAN.md`
   + `docs/BACKEND_API_ORCHESTRATION.md` (moved out of the git-ignored repo root into
   `docs/`) so the branch is self-describing.

## Execution

Use `.venv` for everything; never bare `python`/`python3`. Stage explicit paths on every
commit. Hand each surface to `test-author` and only advance when `make test-unit` is green.

### Step 0 — Branch + baseline

- `git checkout -b feature/backend-api` (off `main`).
- Commit the build-books-index WIP: `git add pyproject.toml Makefile .env.example
  src/extractor/build_books_index.py` → commit "Add build-books-index extractor command".
- Commit artifacts: `git add .claude docs/BACKEND_API_PLAN.md docs/BACKEND_API_ORCHESTRATION.md`
  → commit "Add backend-API agents and plan".
- Confirm baseline green: `make test-unit`.

### Step 1 — Phase 1-2: skeleton + lifespan + deps (api-builder)

Create `src/rag/api/`: `__init__.py`, `app.py` (FastAPI + `lifespan` loading
`config/model/collection/reranker` ONCE onto `app.state`; loud warning — not crash —
if `collection.count()==0`; `run()` → `uvicorn.run` reading `RAG_API_HOST/PORT`;
`if __name__=="__main__": run()` so `python -m rag.api.app` works), `deps.py`
(`get_rag_state(request)`), `auth.py` (stub for now), `schemas.py`, `routes/__init__.py`,
`routes/query.py`, `routes/index.py`. Add `rag-serve = "rag.api.app:run"` to
`[project.scripts]`. Add `fastapi`, `uvicorn[standard]`, `pyjwt` to
`requirements-direct.txt`; recompile `requirements.txt`
(`uv pip compile requirements-direct.txt --python-version 3.10 -o requirements.txt`);
hand-add the three pinned to `requirements-jetson.txt`. Verify `.venv/bin/rag-serve`
boots and app imports. **Commit.**

### Step 2 — Phase 3(health/status) + 5(auth) (api-builder → test-author)

- `auth.py`: HS256 `require_jwt` dependency — reads `Authorization: Bearer <token>`,
  verifies signature + `exp` against `RAG_API_JWT_SECRET` (env only), returns claims or
  401. `GET /health` unauthenticated `{"status":"ok"}`. `GET /status` (JWT) → collection
  name, `collection.count()`, embedding model, reranker model (from app.state, not
  pipeline_status). Optionally a `rag-token` helper to mint a service token.
- **test-author:** `tests/test_api.py` — health no-auth 200; missing/malformed/expired
  token → 401, valid → 200; `/status` reports count + model names. Fake model + temp
  Chroma via dependency override / app.state injection; JWT secret set in a fixture.
- `make test-unit` green → **commit**.

### Step 3 — Phase 3: /query (api-builder → test-author)

- `POST /query` (JWT): Pydantic request `{query, n_results=8, rerank=true, filters:{domain,
  type,source,confidence,subdomain}}`; map non-null filters → `build_where(...)`; call
  `query.search(query, n_results=..., filters=where, config/model/collection from state,
  rerank=...)`; validate `n_results` in **1..50**; response = records + envelope
  `{query, count, reranked}`.
- **test-author:** ranked records; `n_results` cap; filters → build_where; rerank path
  with stub reranker; auth enforced.
- `make test-unit` green → **commit**.

### Step 4 — Phase 4: /index background job (api-builder → test-author)

- `POST /index` (JWT): launch `subprocess.Popen([sys.executable,"-m","rag.indexer", ...])`
  (args as a **list**, never `shell=True`), logs to a file, tracked in an in-process
  registry `{job_id:{status,started,finished,returncode,log_path}}`, status in
  `queued|running|succeeded|failed`. Return **202** + `{job_id}`. `GET /index/jobs/{job_id}`
  returns status. **409** if a job is already `running` (never two indexers on one Chroma
  dir). Nonzero returncode (incl. the 0-files guard's RuntimeError) → job `failed` with
  message. Guard is preserved by delegating to the real indexer — do not bypass.
- **test-author:** 202 + job_id; second concurrent → 409; status transitions; 0-files
  guard surfaces as `failed`. Patch the subprocess launch — no real indexing.
- `make test-unit` green → **commit**.

### Step 5 — Phase 6 compose/Makefile (api-builder) + Phase 7 docs (docs-scribe)

- api-builder: add an `api` service to `docker-compose.yml` AND `docker-compose.jetson.yml`
  — same image, `command: ["python","-m","rag.api.app"]`, `ports: ["8000:8000"]`, same
  volume mounts as `rag`, `RAG_API_JWT_SECRET` via `.env`. Do NOT touch Dockerfile CMD.
  Add `serve` / `jetson-serve` Makefile targets.
- docs-scribe: new `docs/api.md` (all endpoints, JWT flow, token minting, curl + Python
  client snippet); update `README.md`, `CLAUDE.md` (layout tree + "What this does" +
  entry points), `docs/configuration.md` (new env vars in table style), `.env.example`
  (new vars). Note Tailscale-internal / JWT-only / no-in-app-TLS assumption.
- `make test-unit` green → **commit**.

### Step 6 — Review (code-reviewer)

Full read-only review against the plan guardrails + security (JWT env-only + signature +
exp; subprocess not shell-injectable; n_results cap; no secret/path leakage; model loaded
once; CLI + Dockerfile CMD unchanged; deps in both lockfiles; tests cover auth-fail, 409,
0-files guard). Fix blockers via api-builder / test-author. **Then STOP** — do not open the
PR. Summarize changes, test results, and any flags for the user.

## Guardrails (do not violate)

- Reuse `query.get_model/open_collection/get_reranker/build_where/search` — never
  reimplement retrieval.
- Load model+collection+reranker exactly once (lifespan); never per request.
- Indexing never runs synchronously in a request; never two concurrent runs on one
  Chroma dir (409); preserve the indexer 0-files anti-wipe guard.
- CLI entry points (`rag-query`, `rag-index`) and both Dockerfile CMDs unchanged.
- JWT secret from `RAG_API_JWT_SECRET` env only; `/health` open, all else protected.
- `.venv` only; regenerate both lockfiles when adding deps.

## Verification

- After every phase: `make test-unit` (offline, no network, no real model) must pass.
- Skeleton smoke: `.venv/bin/rag-serve` boots; `curl localhost:8000/health` → `{"status":"ok"}`.
- Auth smoke: `/status` without token → 401; with a minted token → 200 + real
  `collection.count()`.
- CLI regression: `.venv/bin/rag-query "kubernetes" -n 3` still returns results
  (proves shared seams unbroken).
- Final: `git diff main...HEAD --stat` + `make test-unit` green before summarizing.
- Do NOT open the PR automatically — stop after code-reviewer is clean and summarize.
