# ADR: Multi-corpus config profiles + pluggable retrieval store

- **Status:** Accepted — Axis 1 (config profiles) and Axis 2 (`RetrievalStore`/`ChromaStore`)
  implemented and merged 2026-07-20/21. Axis 3 (`OpenSearchStore`) remains deferred until
  OpenSearch is actually on the table for the Logmanager wiki chatbot (see rollout step 3 below).
- **Context repo:** personal-rag

## Context

personal-rag is retrieval-only over a single ChromaDB collection bound in `config.yaml`.
Two needs are emerging:

1. Run the same engine in different **sizes**: a light profile for the personal KB on the
   Jetson Orin Nano (8 GB), and a heavier profile for a Logmanager wiki corpus on x86
   servers (no 8 GB ceiling).
2. Keep the door open to swap the retrieval **store** to OpenSearch — the likely
   production target for the Logmanager wiki chatbot, since Logmanager already runs
   OpenSearch.

Verified against the code:

- The engine is already config-driven: `load_config()` honors `RAG_CONFIG_PATH` plus
  `RAG_VAULT_PATH` / `RAG_INDEX_PATH` / `RAG_JSON_PATH` / model / chunk / batch / worker
  overrides.
- Retrieval is already collection-parameterized: `search(..., collection=/collection_name=)`
  and `open_collection(config, name)`.
- ChromaDB is touched in only ~6 places (see call-site map below) — so the store is a
  **bounded** dependency, not woven through the codebase.

## Decision

Treat this as **two independent axes**. They can ship separately.

### Axis 1 — Config profiles (no code)

A "solution" is one **config profile** selected at deploy time via `RAG_CONFIG_PATH`
(or env overrides). A profile controls:

| Knob | Personal (Jetson, light) | Logmanager (x86, heavy) |
|---|---|---|
| `embedding_model` | `all-MiniLM-L6-v2` | larger (e.g. `bge-base`/`e5-large`) |
| `reranker_model` | ms-marco-MiniLM-L-6 | stronger, or off if hurting |
| `rerank_fetch_k` / `tag_fetch_k` | 20 / 200 | higher |
| `embedding_batch_size` | 16 | 64+ |
| `*_workers` | 1 | many |
| `collection_name` | `obsidian_markdown` | `wiki_lm` |
| `vault_path` / sources | personal KB + Books/Resources | wiki repo only |
| `index_path` | `./chroma_db` | its own store dir |

Ship `config.personal.yaml` and `config.logmanager.yaml`. Same codebase; two instances
(each `rag-index` / `rag-serve` with its own `RAG_CONFIG_PATH`). The wiki corpus uses only
the **generic markdown → index path** — the whole book/catalog pipeline (enrich,
build-index, Resource Notes, MOC backlinks, `resource_inventory`) is KB-specific and is
simply not run for the wiki. The markdown extractor already reads any frontmatter schema
via `meta.get(...) or ""`, so WikiJS frontmatter indexes without changes.

**This axis is immediate and near-zero code** — mostly two config files + docs.

### Axis 2 — Pluggable `RetrievalStore` (bounded refactor)

Introduce a `RetrievalStore` protocol; make `search`, `indexer`, and `indexing` depend on
it instead of importing `chromadb` directly. Ship `ChromaStore` (wraps today's behavior,
the default) and — when OpenSearch is actually on the table — `OpenSearchStore`.

```python
class RetrievalStore(Protocol):
    def ensure(self, name, dim, metric): ...      # today: indexer.py:63/73, query.py:66/78
    def existing_ids(self) -> set[str]: ...        # today: indexer.py:80 (incremental diff snapshot)
    def upsert(self, ids, embeddings, docs, metas): ...  # today: indexing.py:27
    def delete(self, ids): ...                     # today: indexer.py:124 (prune stale)
    def query(self, embedding, k, where) -> hits: ...    # today: query.py:133
    def count(self) -> int: ...                    # today: eval.py:81, API /status
```

- `ChromaStore` wraps the existing `PersistentClient` / `get_or_create_collection` /
  `upsert` / `get` / `delete` / `query` / `count` calls — **behavior-preserving**, no
  ranking change, incremental content-hash diff + prune semantics + cosine metric intact.
- `OpenSearchStore` implements the same interface via k-NN + BM25. Because OpenSearch does
  lexical and vector natively, its `query()` can return a **hybrid** result — which would
  **subsume roadmap item 6 (BM25 hybrid)** on that backend.
- Config gains `store: chroma | opensearch` with backend-specific settings nested. Callers
  never import `chromadb` again; they go through the store.

## Consequences

- **Profiles**: light/heavy + a second corpus available now, no engine change.
- **Store abstraction**: ~6 methods across ~4 files (`query.py`, `indexer.py`,
  `indexing.py`, new `store/`); behavior-preserving for Chroma; unlocks OpenSearch without
  a rewrite; OpenSearch backend gets BM25 hybrid for free.
- **Out of scope (unchanged by this ADR):**
  - Generation/LLM layer — retrieval ≠ chatbot. The answer-synthesis step is added *above*
    `search()` (a consumer or a new `/answer` endpoint), identical for both backends.
  - Multi-collection-in-one-service — `search()` already supports it; deferred until needed
    (two instances cover the near term).
- **Risks:** the store refactor must not regress Chroma behavior. Guard with (a) a
  ChromaStore parity test, (b) `make eval` unchanged vs the refreshed baseline, (c) the
  0-files anti-wipe guard preserved (it lives in `indexer.main()`, which now drives the
  store — keep the guard on the store's `existing_ids()`/`count()` signal).

## Rollout (independent, in order)

1. **Config profiles** — `config.personal.yaml` / `config.logmanager.yaml` + docs. Do first; unblocks the wiki PoC immediately.
2. **`RetrievalStore` + `ChromaStore`** — behavior-preserving extraction of the ~6 touchpoints; parity tests + eval baseline unchanged.
3. **`OpenSearchStore`** — later, only when OpenSearch is committed for the wiki. Delivers hybrid natively. Full build spec: `docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md` (index mapping, per-method mapping, the `query()` text/hybrid protocol change, RRF DSL, test strategy). Blocked on a provisioned cluster, not on design.

The generation layer (for an actual chatbot) is a separate track, orthogonal to both axes.
