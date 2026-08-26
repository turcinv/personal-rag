# Bugfix Requirements Document

## Introduction

There is no CLI command to inspect index statistics (total chunk count, breakdown by source type, collection names). When a user wants to verify that all books and resources are indexed — especially after an indexing run on the Jetson — they must guess at internal module import paths and write inline Python. This fails in Docker because the module structure isn't obvious (`rag.config` does not exist; `load_config` lives in `rag.utils`). The project has 16 `rag-*` console scripts but none that exposes index health/stats information. This gap makes it impossible to quickly verify indexing completeness without developer-level knowledge of the codebase internals.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user wants to inspect the total chunk count in the index THEN the system provides no CLI command and the user must write ad-hoc Python using internal imports that are not documented or discoverable

1.2 WHEN a user tries to check the breakdown of indexed chunks by source type (json vs markdown) THEN the system provides no way to obtain this information without materializing all chunk metadata at once via `.snapshot()`, which exceeds the Jetson's 8 GB RAM for large indices (200k+ chunks)

1.3 WHEN a user attempts to verify indexing completeness in Docker (Jetson container) by running inline Python THEN the system fails because the user guesses wrong import paths (e.g. `from rag.config import load_config` instead of the correct `from rag.utils import load_config`)

1.4 WHEN a user wants to see which collections exist in the index THEN the system provides no discoverable CLI interface, forcing them to import `chromadb` directly — which violates the project invariant that chromadb is only imported inside `src/rag/store/`

### Expected Behavior (Correct)

2.1 WHEN a user runs `rag-stats` THEN the system SHALL print the total chunk count for the active collection

2.2 WHEN a user runs `rag-stats` THEN the system SHALL print a breakdown of chunk counts by `source_type` metadata (e.g. json, markdown) and the number of unique source files per type, using paginated iteration that never materializes all metadata at once (memory-safe on Jetson 8 GB)

2.3 WHEN a user runs `rag-stats` in the Jetson Docker container (`docker compose -f docker-compose.jetson.yml run --rm rag rag-stats`) THEN the system SHALL work without requiring knowledge of internal import paths — it is a registered console script like the other 16 `rag-*` commands

2.4 WHEN a user runs `rag-stats --collections` (or equivalent flag) THEN the system SHALL list all collection names present in the index

### Unchanged Behavior (Regression Prevention)

3.1 WHEN existing `rag-*` commands (rag-index, rag-query, rag-serve, etc.) are invoked THEN the system SHALL CONTINUE TO function identically — no change to their behavior or argument interface

3.2 WHEN the store module (`src/rag/store/`) is used by other callers (indexer, query, eval, API) THEN the system SHALL CONTINUE TO provide the same `get_store()`, `list_collection_names()`, and `drop_collection()` interfaces without modification

3.3 WHEN the Jetson runs with its existing config profiles and environment variables THEN the system SHALL CONTINUE TO resolve `RAG_INDEX_PATH`, `RAG_VAULT_PATH`, and other overrides identically for all existing commands

3.4 WHEN `chromadb` is imported THEN it SHALL CONTINUE TO be imported only inside `src/rag/store/` — the new stats command uses the store abstraction layer, not a direct chromadb import
