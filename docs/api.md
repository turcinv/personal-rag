# Backend API

HTTP backend for semantic retrieval over the vault + PDF library. It wraps the
same `rag.query.search` core the `rag-query` CLI uses, but keeps the embedding
model, ChromaDB collection, and cross-encoder reranker resident in one
long-lived process so no request pays the cold-start cost.

## Why a server (load once vs CLI cold-start)

`rag-query` reloads the embedding model and the cross-encoder reranker on every
invocation — several seconds of model load before a single query runs. That is
fine for occasional CLI use, but unworkable for the internal bots (Telegram RAG
Bot, Wiki RAG Chatbot) that issue queries interactively.

The API server loads the model, collection, and reranker **exactly once** at
startup (in the FastAPI `lifespan` handler, via the cached getters in
`rag.query`) and stashes them on `app.state`. Every request reuses those
in-memory objects. On the Jetson this is a deliberate trade-off: one shared
server process holds the model + reranker resident against the 8 GB memory
budget, in exchange for fast per-request latency.

## Deployment assumptions

State these explicitly — they define the security posture:

- **Runs on the Jetson** as one shared, long-lived server process. The model and
  reranker are held resident (the intended trade-off vs the CLI cold-start).
- **Reached over Tailscale** — the server is internal-only, not exposed to the
  public internet. It binds `0.0.0.0:8000` inside its network namespace.
- **JWT is the auth layer.** Every endpoint except `/health` requires a valid
  HS256 bearer token signed with the shared secret.
- **No in-app TLS for now.** Transport confidentiality is left to Tailscale;
  the app speaks plain HTTP. Do not expose the port outside the tailnet.

## Running the server

The server reads `RAG_API_HOST`, `RAG_API_PORT`, and `RAG_API_JWT_SECRET` from
the environment / `.env` (see [Configuration](#configuration)).

### Locally

```bash
make serve
# equivalent to: .venv/bin/rag-serve
```

`rag-serve` starts uvicorn on `RAG_API_HOST:RAG_API_PORT` (defaults
`0.0.0.0:8000`). `RAG_API_JWT_SECRET` must be set or every protected route
returns 500.

### In containers

Both Compose files define an `api` service (same image + mounts as `rag`, so it
reads the same Chroma dir and config) that runs `python -m rag.api.app` and
publishes port `8000`:

```bash
make docker-serve    # x86 / macOS:  docker compose up api
make jetson-serve    # on the Jetson: docker compose -f docker-compose.jetson.yml up api
```

The JWT secret is passed through from the host `.env` (`RAG_API_JWT_SECRET`);
`RAG_API_HOST`/`RAG_API_PORT` have compose-level defaults (`0.0.0.0`/`8000`).

## Authentication

All endpoints except `GET /health` require an `Authorization: Bearer <token>`
header. Tokens are HS256 JWTs signed with the shared secret from
`RAG_API_JWT_SECRET`.

| Condition | Response |
|---|---|
| Valid, unexpired token | route runs |
| Missing/malformed `Authorization` header | `401` `missing bearer token` |
| Expired token | `401` `token expired` |
| Bad signature / any other decode failure | `401` `invalid token` |
| `RAG_API_JWT_SECRET` unset on the server | `500` `server auth not configured` |

An unset secret is a **server misconfiguration** (500), not a client auth
failure (401) — the server can't verify anything without it. The secret and the
token are never logged.

### Minting a service token for the bots

Use `rag-token`. It signs a token with the secret from `RAG_API_JWT_SECRET`
(same variable the server uses) and prints it to stdout:

```bash
RAG_API_JWT_SECRET=<secret> rag-token --subject telegram-bot --expires-days 3650
```

- `--subject` (alias `--sub`) sets the `sub` claim — use the bot's name
  (default `service`).
- `--expires-days` sets the lifetime (default `3650` — ~10 years). These are
  internal service-to-service credentials reachable only over Tailscale, so a
  long expiry is intended.

Mint one token per bot, store it in that bot's own secret store, and send it as
the bearer token on every request.

> **Use a strong secret (≥32 bytes).** PyJWT warns on short HS256 keys. Generate
> one with e.g. `python -c "import secrets; print(secrets.token_urlsafe(48))"`
> and set the same value as `RAG_API_JWT_SECRET` on the server and wherever you
> mint tokens.

## Endpoints

### `GET /health`

Unauthenticated liveness probe. Does not touch the model. Use it for container
health checks and readiness polling.

```
200 OK
{"status": "ok"}
```

### `GET /status`

**Auth required.** Reports the live ChromaDB state so a caller can tell whether
the index is actually populated.

```
200 OK
{
  "collection": "obsidian_markdown",
  "count": 143555,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### `POST /query`

**Auth required.** Semantic retrieval over the once-loaded collection.

Request body:

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | — | **Required.** Whitespace-stripped; empty → `422`. |
| `n_results` | int | `8` | Bounded `1..50` (out of range → `422`). |
| `rerank` | bool | `true` | Cross-encoder reranking; `false` = dense retrieval only. |
| `filters` | object | `null` | Optional metadata constraints (all `$eq`). |

`filters` fields (all optional, all strings): `domain`, `subdomain`, `type`,
`source`, `confidence`. They map to `query.build_where` — omitted/null fields
impose no constraint.

Response body:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Echo of the request query. |
| `count` | int | Number of records returned. |
| `reranked` | bool | `true` only when reranking was requested **and** actually applied. |
| `results` | array | Records, best-first. |

Each record in `results`:

| Field | Type | Notes |
|---|---|---|
| `document` | string | The chunk text. |
| `metadata` | object | `path`, `title`, `heading`, `type`, `domain`, `subdomain`, `status`, `source`, `confidence`, `tags`, `wikilinks`. |
| `distance` | float | Dense similarity distance (lower = closer). |
| `rank` | int | 1-based position. |
| `rerank_score` | float | Present only on reranked results. |

Example:

```json
{
  "query": "How does K3s handle secrets?",
  "count": 2,
  "reranked": true,
  "results": [
    {
      "document": "K3s stores secrets ...",
      "metadata": {"path": "Knowledge/DevOps/K3s.md", "title": "K3s", "type": "Knowledge", "domain": "DevOps"},
      "distance": 0.41,
      "rank": 1,
      "rerank_score": 7.82
    }
  ]
}
```

### `POST /answer`

**Auth required.** Retrieval-augmented generation: runs the same retrieval as
`/query`, then asks the configured LLM for a grounded, cited answer. The
generation layer sits *above* `search()` and is orthogonal to the retrieval
store (see the ADR) — the same endpoint works for any store backend.

**Only available when generation is configured.** If the active config has no
`generation` block, or the provider API-key env var is unset, the server starts
normally, `/query` works, and this endpoint returns `503`. See
[Configuration](#configuration) for the `generation` block.

Request body (retrieval knobs mirror `/query`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `query` | string | — | **Required.** Whitespace-stripped; empty → `422`. |
| `n_results` | int | `8` | Chunks fed to the LLM as context. Bounded `1..20` (tighter than `/query`'s 50 to bound prompt/cost). |
| `rerank` | bool | `true` | Same cross-encoder rerank as `/query`. |
| `filters` | object | `null` | Same shape as `/query` (`domain`, `subdomain`, `type`, `source`, `confidence`, `status`, `tags`). |
| `max_tokens` | int | `null` | Override the configured generation `max_tokens` for this call (`1..4096`). |
| `temperature` | float | `null` | Override the configured `temperature` (`0.0..2.0`). |

Response body:

| Field | Type | Notes |
|---|---|---|
| `query` | string | Echo of the request query. |
| `answer` | string | The generated answer, with inline `[n]` citations. |
| `grounded` | bool | `false` when retrieval found nothing — the LLM is **not** called and a fixed "no relevant context" answer is returned (no hallucination). |
| `provider` | string | `anthropic` / `openai`. |
| `model` | string | Model that actually served the request. |
| `reranked` | bool | As in `/query`. |
| `citations` | array | One entry per context passage, `n` matching the `[n]` markers. |
| `usage` | object | Raw provider token-usage (when available); `null` otherwise. |

Each citation: `n` (1-based), `title`, `path`, `domain`, `distance`,
`rerank_score` (when reranked). The citation index `n` lines up with the
passage numbering the model was given, so `[1]` in the answer maps to
`citations[0]`.

Failure modes: `503` (generation not configured), `502` (the upstream LLM call
failed — transport, non-2xx, or an unparseable body), `422` (bad request body).

```json
{
  "query": "How does K3s handle secrets?",
  "answer": "K3s stores secrets in its embedded datastore [1]; they are not encrypted at rest by default [1].",
  "grounded": true,
  "provider": "anthropic",
  "model": "claude-sonnet-5",
  "reranked": true,
  "citations": [
    {"n": 1, "title": "K3s", "path": "Knowledge/DevOps/K3s.md", "domain": "DevOps", "rerank_score": 7.82}
  ],
  "usage": {"input_tokens": 1200, "output_tokens": 180}
}
```

### `POST /index`

**Auth required.** Kicks off a reindex (`python -m rag.indexer`) as a background
subprocess and returns immediately. The command is fixed — no request input is
passed to the subprocess.

```
202 Accepted
{"job_id": "9f1c...", "status": "running"}
```

Only one index run may be active at a time (never two against one Chroma dir). A
second concurrent `POST /index` returns:

```
409 Conflict
{"detail": "an index run is already in progress"}
```

The subprocess inherits the server's environment (same Chroma dir, same
0-files anti-wipe guard in `indexer.main()`). A run that aborts — including the
anti-wipe guard tripping — surfaces as a `failed` job, not a wiped collection.

### `GET /index/jobs/{job_id}`

**Auth required.** Returns the tracked status for a job. `404` if `job_id` is
unknown. Server-side log paths are never included in the body.

```
200 OK
{
  "job_id": "9f1c...",
  "status": "succeeded",
  "started": "2026-07-18T10:00:00+00:00",
  "finished": "2026-07-18T10:06:12+00:00",
  "returncode": 0,
  "error": null
}
```

`status` is one of `running`, `succeeded`, `failed` (a launch failure records
`failed` immediately). On failure, `error` carries a short reason (the tail of
the indexer log, e.g. the anti-wipe `RuntimeError` message) and `returncode` the
process exit code.

## Examples

Set a base URL and token first. Over Tailscale the host is the Jetson's tailnet
name (shown here as `gpu-01`):

```bash
export RAG_API=http://gpu-01:8000
export TOKEN=$(RAG_API_JWT_SECRET=<secret> rag-token --subject my-bot)
```

### curl

```bash
# liveness (no auth)
curl -s $RAG_API/health

# index status
curl -s $RAG_API/status -H "Authorization: Bearer $TOKEN"

# query (reranked, top 5, DevOps only)
curl -s -X POST $RAG_API/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does K3s handle secrets?", "n_results": 5, "filters": {"domain": "DevOps"}}'

# trigger a reindex, then poll the job
curl -s -X POST $RAG_API/index -H "Authorization: Bearer $TOKEN"
curl -s $RAG_API/index/jobs/9f1c... -H "Authorization: Bearer $TOKEN"
```

### Python client

A minimal client the Telegram / Wiki bots can lift directly (`pip install
requests` in the bot's own environment — it is not a dependency of this repo):

```python
import time
import requests

BASE_URL = "http://gpu-01:8000"   # Jetson over Tailscale
TOKEN = "<service token from rag-token>"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def query(text, n_results=8, rerank=True, **filters):
    """Semantic query. Pass filters as kwargs, e.g. domain='DevOps'."""
    body = {"query": text, "n_results": n_results, "rerank": rerank}
    if filters:
        body["filters"] = filters
    resp = requests.post(f"{BASE_URL}/query", json=body, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()["results"]


def reindex(poll_every=5.0, timeout=1800):
    """Trigger a reindex and block until it finishes. Returns the final job record."""
    resp = requests.post(f"{BASE_URL}/index", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    job_id = resp.json()["job_id"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        job = requests.get(
            f"{BASE_URL}/index/jobs/{job_id}", headers=HEADERS, timeout=30
        ).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(poll_every)
    raise TimeoutError(f"index job {job_id} did not finish in {timeout}s")


if __name__ == "__main__":
    for hit in query("How does K3s handle secrets?", n_results=5, domain="DevOps"):
        meta = hit["metadata"]
        print(f"{hit['rank']}. {meta.get('title')} — {meta.get('path')}")
        print(hit["document"][:300], "\n")
```

## Configuration

Three environment variables control the server (documented in the reference
table in [configuration.md](configuration.md); only the API server needs them):

| Variable | Default | Notes |
|---|---|---|
| `RAG_API_JWT_SECRET` | — (required) | HS256 shared secret. No default. Use ≥32 bytes. |
| `RAG_API_HOST` | `0.0.0.0` | Bind address for uvicorn. |
| `RAG_API_PORT` | `8000` | Bind port for uvicorn. |

### Enabling `/answer` (generation)

`/answer` is off unless the active config carries a `generation` block **and**
the provider API key is exported. Add to the config profile:

```yaml
generation:
  provider: anthropic          # anthropic | openai
  model: claude-sonnet-5       # provider model id
  max_tokens: 1024
  temperature: 0.0
  timeout: 60
  api_key_env: ANTHROPIC_API_KEY   # optional; provider default otherwise
  # base_url: https://api.anthropic.com   # optional; OpenAI-compat servers (vLLM, Ollama, LM Studio)
```

Then export the key (never committed):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."     # or OPENAI_API_KEY for provider: openai
```

The key is read from the environment at startup, not from config. An unknown
`provider` is a hard startup error; a missing key just leaves `/answer` at
`503` while `/query` runs. Switching provider/model needs a restart (the
generator is built once in the lifespan).

## See also

- [architecture.md](architecture.md) — indexing pipeline, chunk IDs, ChromaDB state
- [configuration.md](configuration.md) — full `config.yaml` reference and env vars
- [jetson.md](jetson.md) — Jetson install, Docker, memory budget
