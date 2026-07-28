# Backend API — Implementation Plan (for Claude Code)

> **Status: IMPLEMENTED — merged 2026-07-18 (`8fa0b71`). ARCHIVED 2026-07-28.**
> The backend this plans is live under `src/rag/api/`. Everything below is written
> in future tense but is already built; read it for design rationale only, and treat
> the code as authoritative wherever the two disagree. Current API reference:
> [`docs/api.md`](../api.md). Known divergences and one explicitly rejected
> suggestion are listed in [`docs/archive/README.md`](README.md).

**Goal:** Add an HTTP backend (FastAPI) to `personal-rag` so network clients (Telegram
RAG Bot, Logmanager Wiki RAG Chatbot) can call retrieval — and trigger indexing —
without shelling out to the `rag-query` CLI. The CLI stays; the API is an additional
entry point that reuses `query.py:search()` and `indexer.main()`.

**Branch:** `feature/backend-api` (branch off `main`).

**Decided scope:** Query + indexing endpoints, JWT auth.

---

## Why an API (design intent — keep this in mind)

- The whole point vs. CLI is **load the embedding model, Chroma collection, and
  cross-encoder reranker ONCE at process startup** and share them across requests.
  A per-request cold start (what the CLI does) is the thing we're eliminating.
- `search()` already accepts `model=`, `collection=`, `config=`, `rerank=`, `filters=`
  — so the app just resolves those once in a FastAPI `lifespan` and passes them in.
- Indexing is **heavy and long-running** (full re-embed on the Jetson takes minutes).
  It must NOT block the event loop or run synchronously in the request. Run it as a
  background job and return a job id immediately.

---

## Phase 1 — Package skeleton

Create `src/rag/api/` with:

```
src/rag/api/
├── __init__.py
├── app.py          # FastAPI app + lifespan (loads model/collection/reranker once)
├── auth.py         # JWT verification dependency
├── schemas.py      # Pydantic request/response models
├── deps.py         # shared dependencies (settings, app state accessors)
└── routes/
    ├── __init__.py
    ├── query.py    # POST /query, GET /health, GET /status
    └── index.py    # POST /index, GET /index/jobs/{job_id}
```

- Add entry point in `pyproject.toml` `[project.scripts]`:
  `rag-serve = "rag.api.app:run"` where `run()` calls `uvicorn.run(...)` reading
  host/port from config/env.
- FastAPI + uvicorn are already resolved transitively in `requirements.txt`
  (via chromadb). Add them to `requirements-direct.txt` as **explicit** direct deps
  (`fastapi`, `uvicorn[standard]`, `pyjwt`) and regenerate the lockfiles
  (`requirements.txt` and `requirements-jetson.txt`) per the repo's existing
  compile workflow. `pyjwt` is a new direct dep — confirm it lands in both lockfiles.

## Phase 2 — App state & lifespan (the core value)

In `app.py`, use FastAPI `lifespan`:

- On startup: `config = load_config()`; `model = query.get_model(...)`;
  `collection = query.open_collection(config)`; `reranker = query.get_reranker(...)`
  (import the existing cached getters from `query.py` — don't duplicate).
- Stash them on `app.state` (e.g. `app.state.rag = {...}`).
- Provide a `deps.get_rag_state(request)` dependency so routes read them.
- Guard: if the collection is empty at startup, log a loud warning but still serve
  (mirrors the indexer's "0 chunks" caution — an empty collection is a real state,
  not a crash).

## Phase 3 — Query endpoint

`POST /query` (JWT-protected):

Request (Pydantic, `schemas.py`):
```json
{
  "query": "how to configure k3s on Jetson",
  "n_results": 8,
  "rerank": true,
  "filters": { "domain": "DevOps", "type": null, "source": null,
               "confidence": null, "subdomain": null }
}
```
- Map `filters` → `query.build_where(**filters)` (only pass non-null keys).
- Call `query.search(query, n_results=..., filters=where, config=state.config,
  model=state.model, collection=state.collection, rerank=...)`.
- Response: list of records `{document, metadata, distance, rank}` (search's native
  shape) plus a top-level `{query, count, reranked}` envelope.
- Validation: cap `n_results` (e.g. 1..50) to protect the Jetson.

`GET /health` — liveness only (no auth): `{"status": "ok"}`. Cheap, for probes.

`GET /status` (JWT-protected) — reuse logic from `pipeline_status.py` where possible;
return collection name, chunk count (`collection.count()`), embedding model, and
reranker model. This is the "is the index actually populated" check.

## Phase 4 — Indexing endpoint (background job)

`POST /index` (JWT-protected) kicks off a reindex. Because `indexer.main()` is
argparse/CLI-shaped and long-running:

- Run it **out of the request path**. Two acceptable approaches — pick the simpler:
  1. **Subprocess** (recommended): `subprocess.Popen([sys.executable, "-m",
     "rag.indexer", ...])`, capture logs to a file, track by job id. Cleanest
     isolation; a crashed index run can't take down the API; picks up the indexer's
     own 0-files guard for free.
  2. FastAPI `BackgroundTasks` / a threadpool calling a refactored
     `indexer.run(config)` — only if you first extract a callable `run()` out of
     `main()` (leave `main()` as the CLI wrapper). Heavier refactor.
- Keep an in-memory job registry `{job_id: {status, started, finished, returncode,
  log_path}}` (dict is fine — single process). `status` in
  `queued|running|succeeded|failed`.
- **Concurrency guard:** refuse a new index run (409) if one is already `running`.
  Never run two indexers against the same Chroma dir at once.
- Return `202 Accepted` with `{job_id}`. `GET /index/jobs/{job_id}` returns status.
- ⚠️ Preserve the indexer's existing safety guard: it aborts (RuntimeError, no
  pruning) if every source reports 0 files while the collection is non-empty — this
  is the guard from the 2026-07-15 Jetson wipe incident. Don't bypass it; surface
  that failure as job `failed` with the message.

## Phase 5 — JWT auth

`auth.py`:
- HS256, shared secret from env `RAG_API_JWT_SECRET` (never hardcode; add to
  `.env.example`, document in `docs/configuration.md`).
- FastAPI dependency `require_jwt` that reads the `Authorization: Bearer <token>`
  header, verifies signature + `exp`, returns claims (or 401).
- Provide a tiny helper CLI or documented `pyjwt` snippet to mint a token for the
  bots (subject + long expiry for internal service-to-service). Add a `rag-token`
  helper entry point OR document the one-liner in `docs/`.
- `/health` stays unauthenticated; everything else requires a valid token.

## Phase 6 — Containerization & deploy

- Add an `api` service to `docker-compose.jetson.yml` (and the x86 compose):
  same image, `command: ["python", "-m", "rag.api.app"]` (or `rag-serve`),
  `ports: ["8000:8000"]`, mounting the same volumes as the `rag` service so it
  reads the same Chroma dir and config. Pass `RAG_API_JWT_SECRET` via `.env`.
- Do NOT change the existing `Dockerfile.jetson` CMD (keep `rag.query --help` as the
  default); the compose `command:` selects the server. This keeps CLI usage intact.
- Jetson memory note: the API process holds model + reranker resident. That's the
  intended trade-off, but confirm it fits alongside Chroma in the 8 GB budget
  (see docs/jetson.md memory section). One shared server process, not per-client.
- Add `RAG_API_HOST` / `RAG_API_PORT` config/env with sane defaults (0.0.0.0:8000).
- Reachability is via Tailscale (internal), so no TLS termination in-app is required
  for now; JWT is the auth layer. Note this assumption in the docs.

## Phase 7 — Tests & docs

- Unit tests `tests/test_api.py` using FastAPI `TestClient` with a **fake model +
  temp Chroma** (reuse the fixtures/pattern from `tests/test_indexing.py`):
  - `/query` returns ranked records; respects `n_results` cap and filters.
  - auth: missing/invalid/expired token → 401; valid → 200.
  - `/index` returns 202 + job_id; second concurrent call → 409; job status
    transitions; the 0-files guard surfaces as a failed job.
  - `/health` works without auth; `/status` reports collection count.
- Keep it in the offline suite (`make test-unit`) — no network, no real model.
- Docs: new `docs/api.md` (endpoints, auth, examples with curl + a Python client
  snippet the bots can copy). Update `README.md`, `CLAUDE.md` (project layout +
  "What this project does"), and `docs/configuration.md` (new env vars).
- Add a `make serve` / `make jetson-serve` target for local + Jetson runs.

---

## Order of work (commits)

1. Skeleton + deps + `rag-serve` entry point + empty app that boots. (Phase 1–2)
2. `/health` + `/status` + JWT dependency. (Phase 3 health + Phase 5)
3. `/query` end-to-end + tests. (Phase 3 + 7)
4. `/index` background job + concurrency guard + tests. (Phase 4 + 7)
5. Compose service + docs + Makefile targets. (Phase 6 + 7)

Commit after each. Open a PR from `feature/backend-api` when Phase 7 is green.

## Guardrails (do not violate)

- Reuse `query.get_model` / `open_collection` / `get_reranker` / `build_where` /
  `search` — do not reimplement retrieval.
- Load model/collection/reranker exactly once (lifespan), never per request.
- Never run indexing synchronously in a request; never allow two concurrent index
  runs against the same Chroma dir.
- Preserve the indexer's 0-files anti-wipe guard.
- CLI (`rag-query`, `rag-index`) must keep working unchanged.
- Follow the repo's `.venv`-only rule; regenerate both lockfiles when adding deps.
