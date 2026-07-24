# OpenSearchStore — implementation plan (Axis 3)

- **Status:** Planned — not yet implemented. Blocked on a real OpenSearch instance
  (see [Prerequisites](#prerequisites)). This doc is the build spec so the work is
  fast and well-scoped once a cluster exists; it invents no code.
- **Repo:** personal-rag
- **Relates to:** `docs/ADR-multi-corpus-profiles-and-pluggable-store.md` (Axis 2 —
  the store abstraction this slots into), roadmap item 6 (BM25 hybrid — subsumed
  here), and the vault notes *OpenSearch ML Capabilities Research* and *Logmanager
  Wiki RAG Architecture and Governance ADR* (Decision 1: stand up a dedicated
  OpenSearch instance).

## Why now (and why not code yet)

The `RetrievalStore` abstraction (Axis 2) already isolates ChromaDB to
`store/chroma_store.py` — the six-ish touchpoints in `query.py`, `indexer.py`,
`indexing.py`, `eval.py`, and the API go through the store, never `chromadb`
directly. So a second backend is a *bounded* addition: one new module plus a
factory branch, no engine rewrite.

The reason not to write the code today is that `OpenSearchStore`'s entire value
is **native hybrid search (BM25 + k-NN, fused with RRF)** — which can only be
verified against a live OpenSearch cluster with the ML Commons / k-NN plugins.
Writing it blind would produce untestable code that drifts from the real index
mapping and query DSL before a cluster ever exists. The repo's store discipline
(behavior-preserving, parity-tested — see the `store-code-reviewer` subagent)
requires a cluster to test against. This plan removes everything *except* that
cluster from the critical path.

## Prerequisites (from the Wiki RAG ADR, Decision 1 — still open)

Before implementation starts, these must be resolved (they are open questions in
the vault ADR, not decisions):

1. A **dedicated OpenSearch instance** stood up (NOT the product's pinned ES
   7.10.2 fork) on a current version with **k-NN + ML Commons** enabled.
2. Which OpenSearch version — confirm `knn_vector` + `neural`/`hybrid` query and
   the `search-pipeline` RRF processor are all supported on it.
3. Where it is provisioned (which Proxmox/ESX host, disk/RAM), with an owner and a
   ticket.
4. Governance sign-off that the corpus may live on that instance (the wiki corpus
   carries "architecture secrets"; on-prem-only keeps it in policy).

Until (1)–(3) exist, `get_store()` continues to reject `store: opensearch` — no
half-wired backend lands on `main`.

## Design overview

`OpenSearchStore` implements the exact same `RetrievalStore` Protocol
(`store/base.py`) as `ChromaStore`. One store instance owns one index (the
OpenSearch analogue of a Chroma collection). Retrieval, reranking, and rank
assignment stay **above** the store in `query.search()` — the store only fetches
and returns records. The one genuinely new capability is that its `query()` can
fuse BM25 + vector natively, which is where roadmap item 6 gets solved for free
on this backend.

### Dependency

Add `opensearch-py` to `requirements-direct.txt` (and both lockfiles). It is
pure-Python, so it installs cleanly on both x86 and Jetson aarch64 — no wheel
pain like torch. Keep the `import opensearchpy` confined to
`store/opensearch_store.py`, exactly as `chromadb` is confined to
`chroma_store.py` (the ADR's "callers never import a backend client" rule).

## Index mapping

One index per corpus (name from `collection_name`, e.g. `wiki_lm`). Created by
`ensure()` if absent. Proposed mapping:

```jsonc
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 100
    },
    "default_pipeline": null            // ingest-time embedding is out of scope; we embed client-side, same as Chroma
  },
  "mappings": {
    "properties": {
      "chunk_id":  { "type": "keyword" },          // == our stable content-hash id; also the _id
      "document":  { "type": "text" },             // BM25 analyzed field (the chunk body)
      "embedding": {
        "type": "knn_vector",
        "dimension": 384,                           // MiniLM/bge-small = 384; set from config at ensure()
        "method": {
          "name": "hnsw",
          "space_type": "cosinesimil",              // cosine — matches Chroma's hnsw:space=cosine
          "engine": "lucene"
        }
      },
      // metadata fields — all keyword so build_where's $eq maps to term filters:
      "path":       { "type": "keyword" },
      "title":      { "type": "keyword" },
      "heading":    { "type": "keyword" },
      "type":       { "type": "keyword" },
      "domain":     { "type": "keyword" },
      "subdomain":  { "type": "keyword" },
      "status":     { "type": "keyword" },
      "source":     { "type": "keyword" },
      "confidence": { "type": "keyword" },
      "tags":       { "type": "keyword" },          // NOTE: index as a real array, not the Chroma comma-joined string
      "wikilinks":  { "type": "keyword" }
    }
  }
}
```

Two deliberate divergences from Chroma, both improvements the abstraction allows:

- **`tags` is a proper `keyword` array**, not the comma-joined string Chroma 0.6.3
  forced (weak array support). This means tag filtering can be a native `terms`
  filter here (see [build_where](#metadata-filtering-build_where)), retiring the
  post-filter hack from roadmap item 7 *on this backend*. `search()`'s existing
  tag post-filter still works unchanged; the store just needs to expose whether it
  filters tags natively (a small capability flag) so `search()` can skip the
  post-filter to avoid double-filtering. Keep this out of the first pass if it
  complicates parity — the post-filter is correct either way, just redundant.
- **`space_type: cosinesimil`** reproduces Chroma's cosine metric so rankings are
  comparable; embeddings are already L2-normalized on our side.

## Per-method mapping (`RetrievalStore` → opensearch-py)

Every method mirrors `ChromaStore`'s contract exactly (`store/base.py` is the
spec). `client` below is an `OpenSearch(...)` from `opensearchpy`.

| Method | ChromaStore today | OpenSearchStore |
|---|---|---|
| `name` | collection name | index name |
| `ensure(name, dim, metric)` | `get_or_create_collection(hnsw:space)` | `indices.exists` → if absent `indices.create` with the mapping above; **`dim` is now USED** (sets `knn_vector.dimension`) unlike Chroma which ignores it; `metric`→`space_type`. Idempotent: never recreate an existing index. |
| `snapshot()` | `get(include=["metadatas"])` → `{id: meta}` | scroll/`search` over the whole index requesting only metadata fields (`_source` minus `embedding`/`document`), build `{_id: {meta}}`. Use `point-in-time` + `search_after` (or the scroll API) so it scales past 10k docs. |
| `existing_ids()` | `get(include=[])["ids"]` | same scroll, `_source: false`, collect `_id`s into a set. |
| `count()` | `collection.count()` | `client.count(index)["count"]`. |
| `upsert(ids, embeddings, docs, metas)` | `collection.upsert(...)` | `helpers.bulk` with `_op_type: index` (index == upsert-by-id), `_id=chunk_id`, `_source={document, embedding, **meta}`. Chunk with `refresh=False` for speed; one `indices.refresh` after the batch. |
| `update_metadata(ids, metas)` | `collection.update(metadatas=)` | `helpers.bulk` with `_op_type: update`, `doc={**meta}` — metadata-only, embedding untouched (preserves the "no re-embed on frontmatter-only change" optimization). |
| `delete(ids)` | `collection.delete(ids)` | `helpers.bulk` with `_op_type: delete` (batched; the indexer already chunks stale IDs at 500). |
| `query(embedding, k, where, ...)` | k-NN only | dense k-NN **or** hybrid — see below. **Interface change required.** |

### The one interface change: `query()` needs the raw text

`RetrievalStore.query(embedding, k, where)` currently has **no text argument** —
fine for pure vector search, impossible for BM25. To enable hybrid, extend the
Protocol (and both stores) to:

```python
def query(self, embedding, k, where=None, *, text=None, hybrid=False) -> list: ...
```

- `ChromaStore` ignores `text`/`hybrid` (Chroma has no BM25) — behavior-preserving,
  so parity + eval stay byte-identical.
- `OpenSearchStore` uses `text` for the BM25 clause when `hybrid=True`, else runs
  pure k-NN.
- `query.search()` already threads a `hybrid=False` flag it never wired
  (`query.py`); this is where it finally connects. `search()` passes the raw query
  string (it already has it — the same text used for the reranker, never the
  embedding-prefixed variant) down as `text=`.

Records returned keep the exact shape `search()` expects: `{"document",
"metadata", "distance"}`, best-first. **Score→distance mapping:** OpenSearch
`_score` is higher-is-better; our record `distance` is lower-is-better. Return
`distance = 1.0 - normalized_score` (or `-_score`) purely so the field stays
monotonic for display/debug — `search()` does **not** re-sort dense/hybrid
results (it trusts store order and only re-sorts when the cross-encoder reranks),
so the exact value is cosmetic. Document this so nobody later assumes it's a true
cosine distance.

## Metadata filtering (`build_where`)

`build_where()` returns Chroma's `$eq`/`$and` dict today. Two clean options:

1. **Translate in the store** (recommended): `OpenSearchStore.query()` converts the
   Chroma-style where-dict into an OpenSearch `bool.filter` of `term`/`terms`
   clauses. Keeps `build_where` and every caller unchanged; the translation is
   ~20 lines and fully unit-testable without a cluster.
2. Add a `build_filter` seam per backend. More invasive; only worth it if the
   dialects diverge more later. Not needed now.

`status` → `term`. `tags` → `terms` (native array match — the item-7 win). The
`$and` list → `bool.filter` array.

## Hybrid query DSL (BM25 + k-NN + RRF)

Pure k-NN (`hybrid=False`):

```jsonc
{
  "size": k,
  "query": { "knn": { "embedding": { "vector": <emb>, "k": k } } },
  "filter": { "bool": { "filter": [ <term/terms from where> ] } }
}
```

Hybrid (`hybrid=True`) via the **`hybrid` query + a search pipeline** doing
score normalization + combination (OpenSearch's native RRF-style fusion):

```jsonc
// one-time: create a search pipeline (in ensure(), or a separate provisioning step)
PUT /_search/pipeline/hybrid-rrf
{
  "phase_results_processors": [
    { "normalization-processor": {
        "normalization": { "technique": "min_max" },
        "combination": { "technique": "arithmetic_mean",
                         "parameters": { "weights": [0.3, 0.7] } } } }  // [bm25, knn] — tune in eval
  ]
}

// query:
{
  "size": k,
  "query": {
    "hybrid": {
      "queries": [
        { "match": { "document": { "query": <text> } } },              // BM25
        { "knn":   { "embedding": { "vector": <emb>, "k": k } } }       // dense
      ]
    }
  },
  "post_filter": { "bool": { "filter": [ <term/terms> ] } }             // or fold into each sub-query
}
```

Notes:
- The fetch width (`k`) should honor the existing `rerank_fetch_k` / `tag_fetch_k`
  logic in `search()` — the store just receives the final `k`. Hybrid pairs
  naturally with the cross-encoder reranker already in `search()`: retrieve a broad
  hybrid candidate set, rerank down. This is exactly the "hybrid + rerank ≈ +9 MRR"
  stack the Wiki RAG ADR's Decision 6 points at.
- Weights `[0.3, 0.7]` (bm25/knn) are a starting guess — **tune against the eval
  harness**, don't hardcode as final.

### This subsumes roadmap item 6

Roadmap item 6 (BM25/lexical hybrid) is listed as "highest impact, highest effort,
possibly a store change." On OpenSearch it is not extra work — it *is* the
`hybrid=True` path above. When this backend lands, mark item 6 done **for the
OpenSearch backend** (Chroma still has no BM25 and doesn't need it — MiniLM dense
+ rerank already scores well on the personal corpus per `tests/eval/baseline.json`).

## Config

`get_store()` gains an `opensearch` branch:

```python
backend = config.get("store", "chroma")
if backend == "chroma":
    return ChromaStore(index_path, name)
if backend == "opensearch":
    return OpenSearchStore(config["opensearch"], name)
raise ValueError(...)
```

`config.logmanager.yaml` (the profile that would use it) gains:

```yaml
store: opensearch
opensearch:
  hosts: ["https://opensearch.internal:9200"]
  # auth via env, never in the file (mirrors the generation api_key_env pattern):
  user_env: RAG_OPENSEARCH_USER
  password_env: RAG_OPENSEARCH_PASSWORD
  verify_certs: true
  ca_certs: /path/to/ca.pem
  hybrid: true                 # default the search() hybrid flag on for this corpus
  hybrid_weights: [0.3, 0.7]   # [bm25, knn]
  search_pipeline: hybrid-rrf
```

Credentials come from env vars named in the config, not the file — same rule as
the generation layer's `api_key_env`.

## Test strategy

Two tiers, mirroring the existing store tests:

1. **Offline unit tests (no cluster)** — the majority, and enough to catch most
   regressions:
   - where-dict → OpenSearch `bool.filter` translation (pure function; table of
     cases incl. `$and`, `status` term, `tags` terms, empty→no filter).
   - query-DSL construction for `hybrid=False` vs `True` (assert the JSON body
     shape, k, filter placement) using a **mocked** `OpenSearch` client that
     records the `search(body=...)` it was handed — same spirit as the
     `httpx.MockTransport` approach used for the generation layer.
   - `upsert`/`update_metadata`/`delete` build the right `helpers.bulk` actions.
   - score→distance monotonicity mapping.
2. **Integration tests (real cluster, opt-in)** — gated behind an env flag /
   pytest marker so CI's offline suite stays offline. Use a **dockerized
   OpenSearch** (`opensearchproject/opensearch:<ver>` via testcontainers-python or
   a compose service) to verify: `ensure` creates the mapping; upsert→query
   round-trips; k-NN returns nearest; hybrid returns fused results; metadata
   filters apply; `snapshot`/`existing_ids`/`count` reflect the real population;
   the incremental diff (new/changed/unchanged/prune) behaves like Chroma.
   - **Parity check:** index the same small fixture into both a temp Chroma and the
     docker OpenSearch, assert `query()` returns the same *set* of top-k ids for
     dense-only (order may differ slightly across engines — assert set/overlap, not
     exact order). This is the behavior-preserving gate.
   - Extend `tests/eval/` with a `--store opensearch` path so recall@k / MRR can be
     measured on the real backend and compared to the Chroma baseline before
     trusting it.

## Rollout order

1. Prereqs (1)–(4) above — **the actual blocker.**
2. Protocol change: add `text`/`hybrid` kwargs to `RetrievalStore.query` +
   `ChromaStore` (no-op) + `search()` threading. Ship this *first*, behind the
   existing `hybrid=False` default — it's behavior-preserving and independently
   reviewable, and unblocks the store work.
3. `OpenSearchStore` + `get_store` branch + config + offline unit tests.
4. Integration tests against the docker/real cluster; tune hybrid weights on eval.
5. Point `config.logmanager.yaml` at the instance; index the wiki corpus; compare
   eval vs Chroma baseline; mark roadmap item 6 done for this backend.

## Open questions (carry into implementation)

- Native tag `terms` filtering vs. keeping `search()`'s post-filter — dedupe to
  avoid double-filtering, or leave the (correct, redundant) post-filter for the
  first pass?
- Client-side embedding (as today) vs. OpenSearch ingest-pipeline ML inference
  (embed inside the cluster). The vault ADR's Decision 5 leans toward the
  ingest-pipeline pattern for a continuously-synced index; this plan assumes
  client-side embedding first (smallest change, reuses the existing embed path)
  and treats ingest-pipeline embedding as a later optimization.
- Exact fusion technique + weights — start with `min_max` + weighted arithmetic
  mean, tune on eval; revisit RRF-rank fusion if score normalization underperforms.
