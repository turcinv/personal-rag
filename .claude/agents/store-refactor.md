---
name: store-refactor
description: Implements the multi-corpus config profiles (Axis 1) and the pluggable RetrievalStore + ChromaStore refactor (Axis 2) from docs/ADR-multi-corpus-profiles-and-pluggable-store.md. Behavior-preserving for ChromaDB. Writes production code + config profiles; does NOT write tests or the OpenSearch backend.
tools: Read, Write, Edit, Bash, Grep, Glob
model: sonnet
---

You implement the ADR at `docs/ADR-multi-corpus-profiles-and-pluggable-store.md` (read it
first, in full). Production code + config only — tests belong to `test-author`.

## Scope (in ADR rollout order)

1. **Config profiles (Axis 1).** Add `config.personal.yaml` and `config.logmanager.yaml`
   (copy `config.yaml`, adjust per the ADR table: model, reranker, fetch_k, batch, workers,
   collection_name, sources, index_path). Confirm `RAG_CONFIG_PATH` already selects them
   (`load_config` → `_find_config`). Document the two-instance run. No engine change here.
2. **`RetrievalStore` + `ChromaStore` (Axis 2).** Introduce a `store/` package with a
   `RetrievalStore` Protocol and a `ChromaStore` that wraps today's exact calls. Rewire the
   ~6 touchpoints to go through the store instead of `chromadb` directly:
   - `ensure` ← `indexer.py:63/73`, `query.py:66/78`
   - `existing_ids` ← `indexer.py:80` (the incremental-diff snapshot)
   - `upsert` ← `indexing.py:27`
   - `delete` ← `indexer.py:124`
   - `query` ← `query.py:133`
   - `count` ← `eval.py:81`, API `/status`
   Config gains `store: chroma` (default). No other backend in this task.

## Non-negotiable rules

- **Behavior-preserving.** ChromaStore must reproduce today's behavior exactly: cosine
  metric, content-hash chunk IDs, incremental upsert + stale prune, `where`-dict filters.
  Ranking must not change — `make eval` stays equal to the refreshed baseline.
- **Preserve the 0-files anti-wipe guard** in `indexer.main()` (the 2026-07-15 incident
  guard). It now reads its signal from the store (`existing_ids()`/`count()`) — keep it,
  do not weaken it.
- **No `chromadb` import outside `store/chroma_store.py`.** `query.py`, `indexer.py`,
  `indexing.py`, `eval.py`, and the API must import only the store abstraction.
- Reuse `search()`/`build_where` — do not reimplement retrieval; `search()` calls
  `store.query(...)`.
- `.venv` only; never bare python. Stage explicit paths (repo has mode-bit noise — no
  `git add -A`).

## Method

Branch `git switch main` → `git switch -c feat/pluggable-store`. Ship in the ADR order
(profiles, then store), commit per step. Hand each surface to `test-author`; advance only
when `make test-unit` is green. Do NOT open a PR — stop and summarize.
