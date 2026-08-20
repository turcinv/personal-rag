# Security & privacy posture

`personal-rag` runs a private knowledge base on a personal workstation and a
Jetson reachable over Tailscale. The design keeps data local, network egress
explicit, and the write path coordinated. This is the consolidated reference for
those properties; mechanics live in the linked docs.

## Network egress

- **Indexing and retrieval never leave the machine.** No cloud storage/compute
  SDK is imported anywhere in `src/rag/` (`boto3`, `google.cloud`, `gsutil` — a
  test, `tests/test_packaging.py`, keeps it that way).
- **Exactly one intentional outbound path:** answer generation
  (`src/rag/generation/`) calls Anthropic/OpenAI (or a local OpenAI-compatible
  server) over `httpx`. It is **off by default** — with no `generation` block and
  no API key, `/answer` returns 503 and `/query` is unaffected. `httpx` is
  confined to the generation layer (also test-enforced).
- **Egress destinations are validated.** A configured `base_url` must be `https`
  for a remote host; plain `http` is allowed only for a local inference server
  (localhost). An invalid/insecure destination fails the server at startup.
- **Provider responses never reach clients.** On a provider error the status code
  is surfaced but the provider body is logged server-side only, never returned in
  the API response.

## Store integrity & the write path

- `chromadb` is confined to `src/rag/store/chroma_store.py`; every other layer
  goes through the `RetrievalStore` seam and `query.search()`.
- A **cross-process writer lock** (`src/rag/locking.py`) serializes the indexer,
  lexical build, backup, and destructive maintenance against one index dir. Stale
  locks self-heal (kernel releases `flock` on process death). See
  [recovery.md](recovery.md).
- **Index provenance** (model/dimension/chunker/metric/profile fingerprint) is
  stored with the collection and checked on open, so a mismatched model can never
  silently mix vectors. Lexical sidecars are bound to the vector generation and
  refuse to serve stale hybrid results.
- Backups are quiesced (taken under the writer lock) and verified by checksum +
  count + provenance; a restore drill runs on a temp path.

## API authorization

- All routes except `/health` and `/ready` require a JWT (`RAG_API_JWT_SECRET`,
  HS256, ≥32 bytes — validated at startup).
- Optional **scopes** (`query`, `answer`, `index`) least-privilege a token; a
  token with no `scopes` claim keeps full access for compatibility. Mint scoped
  tokens with `rag-token --scope query`.
- Optional **audience/issuer** enforcement when `api.jwt_audience`/`jwt_issuer`
  (or the matching env vars) are set.
- Default token lifetime is 30 days (was 3650). A bounded inference-concurrency
  guard protects the Jetson's shared memory.

## Privacy & retention

- **Raw query text is not logged by default** — only its length. Set
  `log_queries: true` to log verbatim queries for debugging.
- The structured SQLite log is pruned on startup to `log_retention_days`
  (default 30) so it cannot grow without bound.

## Offline operation

- `offline: true` (or `RAG_OFFLINE`) forces `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`
  before any model load, so a cold cache fails fast instead of reaching the
  network. Pair with `embedding_revision` to pin an exact model snapshot.

See also: [configuration.md](configuration.md) for every key,
[recovery.md](recovery.md) for the lock/backup/restore workflow, and
[architecture.md](architecture.md) for the retrieval seam and store boundary.
