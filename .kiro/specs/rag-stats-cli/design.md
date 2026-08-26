# rag-stats CLI Bugfix Design

## Overview

The project has 16 `rag-*` console scripts but no way to inspect index health without writing ad-hoc Python against internal imports. This is especially painful on the Jetson (Docker container) where module paths are non-obvious and `.snapshot()` would OOM on 200k+ chunks. The fix adds a `rag-stats` command that exposes total chunk count, per-source-type breakdown, and collection listing through the existing store abstraction — paginated, memory-safe, and following the exact same CLI pattern as `rag-query` and `rag-pipeline-status`.

## Glossary

- **Bug_Condition (C)**: A user wants to inspect index statistics (chunk count, source breakdown, collection list) but no CLI command exists — forcing ad-hoc Python with internal imports that fail in Docker
- **Property (P)**: When `rag-stats` is invoked, it prints accurate statistics using the store abstraction layer with paginated iteration, never exceeding the Jetson's 8 GB memory
- **Preservation**: All existing `rag-*` commands, store interfaces, config resolution, and the chromadb-import invariant remain unchanged
- **`ChromaStore.count()`**: Returns total chunk count without loading data
- **`ChromaStore.iter_records(page_size)`**: Yields `(id, doc, metadata)` in pages — memory-safe on Jetson
- **`list_collection_names(config)`**: Module function in `src/rag/store/` returning all collection names
- **`get_store(config)`**: Factory that opens a `RetrievalStore` bound to one collection

## Bug Details

### Bug Condition

The bug manifests when a user needs to verify indexing completeness or inspect index health. There is no CLI command exposing this information, so users must write inline Python that (a) guesses internal import paths (which differ between editable-install dev and Docker), (b) may call `.snapshot()` which materializes all metadata at once and OOMs on Jetson with 200k+ chunks, or (c) imports `chromadb` directly which violates the project invariant.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type UserAction
  OUTPUT: boolean
  
  RETURN input.intent IN ['check_chunk_count', 'check_source_breakdown',
                          'list_collections', 'verify_indexing_completeness']
         AND NOT existsCLICommand(input.intent)
         AND (userMustWriteAdHocPython(input.intent)
              OR userMustImportChromaDirectly(input.intent))
END FUNCTION
```

### Examples

- User runs `rag-stats` → currently fails with "command not found" because the script does not exist
- User tries `python -c "from rag.config import load_config; ..."` in Docker → ImportError because the module is `rag.utils`, not `rag.config`
- User calls `.snapshot()` on a 200k-chunk index on Jetson → process killed by OOM (materializes all metadata dicts at once into 8 GB unified RAM)
- User imports `chromadb` in a script outside `src/rag/store/` to call `collection.count()` → violates the store-confinement invariant enforced by `tests/test_store.py`

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- All 16 existing `rag-*` commands (`rag-index`, `rag-query`, `rag-serve`, `rag-eval`, etc.) continue to function identically with no argument or output changes
- `get_store()`, `list_collection_names()`, and `drop_collection()` interfaces in `src/rag/store/__init__.py` remain unmodified
- Config resolution (YAML + env-var overrides for `RAG_INDEX_PATH`, `RAG_VAULT_PATH`, etc.) works identically for all existing commands
- `chromadb` continues to be imported only inside `src/rag/store/` — the new stats command uses `get_store()` and `list_collection_names()`, never a direct chromadb import

**Scope:**
All inputs that do NOT involve the new `rag-stats` command should be completely unaffected by this fix. This includes:
- Indexing (`rag-index`, `make index`)
- Querying (`rag-query`, `POST /query`)
- Serving (`rag-serve`)
- Evaluation (`rag-eval`, `make eval`)
- All extractor commands (`rag-extract`, `rag-enrich`, etc.)

## Hypothesized Root Cause

This is a missing-feature bug rather than a logic defect. The root cause is straightforward:

1. **No `src/rag/stats.py` module exists**: The statistics logic has not been written
2. **No console script entry point**: `pyproject.toml` does not register `rag-stats`
3. **No Makefile targets**: Neither `make stats` nor `make jetson-stats` exist
4. **Existing store API is sufficient**: `ChromaStore` already has `.count()` and `.iter_records(page_size)` — the building blocks exist but are not wired to a user-facing command

## Correctness Properties

Property 1: Bug Condition - Stats Command Produces Accurate Output

_For any_ invocation of `rag-stats` (with or without `--collections`) against an index containing N chunks across S source types, the command SHALL print the correct total count (matching `store.count()`), correct per-source-type breakdown (counts summing to N), and correct unique file counts per type, all obtained via paginated iteration that never holds more than one page of metadata in memory.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Preservation - Existing Commands and Interfaces Unchanged

_For any_ invocation of an existing `rag-*` command or programmatic use of the store module (`get_store`, `list_collection_names`, `drop_collection`), the system SHALL produce exactly the same behavior as before the fix — no interface changes, no import changes, no config resolution changes, and no new chromadb imports outside `src/rag/store/`.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

## Fix Implementation

### Changes Required

**File**: `src/rag/stats.py` (new)

**Function**: `main()`

**Specific Changes**:

1. **Create `src/rag/stats.py`**: New module following the established CLI pattern:
   - `argparse` for argument parsing (optional `--collections` flag, optional `--page-size` for tuning)
   - `load_config()` from `.utils`
   - `setup_logging(config, console=False)` (log to file, stats print to stdout)
   - `get_store(config)` and `list_collection_names(config)` from `.store`
   - `store.count()` for total chunk count
   - `store.iter_records(page_size)` to accumulate per-source-type stats without full materialization

2. **Paginated aggregation logic**: Iterate records in pages (default 10,000), for each chunk read `metadata.get("source_type", "unknown")` and `metadata.get("path", "")`, accumulate:
   - Count per source_type
   - Set of unique paths per source_type
   - Never hold more than one page of `(id, doc, metadata)` tuples in memory

3. **Output formatting**: Print human-readable stats to stdout:
   - Total chunks (from `.count()` — O(1), no iteration needed)
   - Per source_type: chunk count + unique file count
   - When `--collections` flag is set: list all collection names via `list_collection_names(config)`

4. **Register console script**: Add `rag-stats = "rag.stats:main"` to `pyproject.toml` `[project.scripts]`

5. **Makefile targets**: Add `make stats` (runs `rag-stats`) and `make jetson-stats` (runs via docker-compose)

### Pseudocode

```python
def main():
    parser = argparse.ArgumentParser(description="Index statistics")
    parser.add_argument("--collections", action="store_true",
                        help="List all collection names in the index")
    parser.add_argument("--page-size", type=int, default=10_000,
                        help="Iteration page size (tune for memory)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config, console=False)

    if args.collections:
        names = list_collection_names(config)
        print(f"Collections ({len(names)}):")
        for name in sorted(names):
            print(f"  {name}")
        return

    store = get_store(config)
    total = store.count()
    print(f"Collection: {store.name}")
    print(f"Total chunks: {total:,}")

    # Paginated aggregation — never materializes full index
    type_counts: dict[str, int] = {}
    type_files: dict[str, set[str]] = {}

    for _id, _doc, meta in store.iter_records(page_size=args.page_size):
        st = meta.get("source_type", "unknown")
        type_counts[st] = type_counts.get(st, 0) + 1
        path = meta.get("path", "")
        if path:
            type_files.setdefault(st, set()).add(path)

    print(f"\nBy source_type:")
    for st in sorted(type_counts):
        count = type_counts[st]
        files = len(type_files.get(st, set()))
        print(f"  {st:12s}  {count:>7,} chunks  ({files:,} files)")
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, confirm the bug exists (no `rag-stats` command), then verify the fix works correctly and existing behavior is preserved.

### Exploratory Bug Condition Checking

**Goal**: Confirm the bug exists BEFORE implementing the fix — there is no `rag-stats` command and users must resort to internal imports.

**Test Plan**: Attempt to run `rag-stats` from the shell and confirm it fails. Attempt the documented workarounds and observe the failure modes.

**Test Cases**:
1. **Command Missing Test**: Run `rag-stats` — will fail with "command not found" on unfixed code
2. **Wrong Import Path Test**: Run `python -c "from rag.config import load_config"` — will fail with ImportError
3. **Snapshot OOM Test**: On a large index, calling `.snapshot()` would exceed memory (documented, not safe to exercise in test)
4. **Direct chromadb Import Test**: Any script importing chromadb outside `src/rag/store/` would fail `test_store.py`'s invariant check

**Expected Counterexamples**:
- `rag-stats` is not a registered console script
- Users guess `rag.config` instead of `rag.utils` for `load_config`

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed function produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := rag_stats(input)
  ASSERT result.total == store.count()
  ASSERT sum(result.type_counts.values()) == result.total
  ASSERT result.exit_code == 0
  ASSERT peak_memory < page_size * avg_record_size  (not full index)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed function produces the same result as the original function.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT existing_commands_unchanged(input)
  ASSERT store_interface_unchanged(input)
  ASSERT config_resolution_unchanged(input)
  ASSERT chromadb_import_confinement_holds()
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It can generate random config profiles and verify `get_store()` still resolves identically
- It can verify the chromadb-import invariant across the entire `src/` tree
- It catches accidental regressions in config override logic

**Test Plan**: Observe behavior on UNFIXED code first (existing tests pass), then write property-based tests capturing that existing behavior continues after the new module is added.

**Test Cases**:
1. **Store Interface Preservation**: Verify `get_store()`, `list_collection_names()`, `drop_collection()` signatures and return types are unchanged
2. **Config Resolution Preservation**: Verify `load_config()` with various env-var overrides produces identical results
3. **Import Invariant Preservation**: Verify `chromadb` is not imported anywhere outside `src/rag/store/` (existing `test_store.py` check)
4. **Existing CLI Preservation**: Verify `rag-query --help`, `rag-index --help` still work identically

### Unit Tests

- Test `main()` with a mocked store returning known metadata — verify output format and counts
- Test `--collections` flag with mocked `list_collection_names` returning known names
- Test with empty index (0 chunks) — verify graceful output
- Test with unknown `source_type` values in metadata — verify they appear as-is
- Test `--page-size` argument is passed through to `iter_records`
- Test that no chromadb import exists in `src/rag/stats.py`

### Property-Based Tests

- Generate random collections of metadata dicts with varied `source_type` and `path` values, feed through the aggregation logic, verify counts always sum to total and unique file sets are correct
- Generate random page sizes and verify aggregation produces identical results regardless of page size
- Generate random config dicts and verify `get_store` / `list_collection_names` dispatch is unchanged

### Integration Tests

- Run `rag-stats` against a real (test) ChromaDB index with known contents and verify output matches expected counts
- Run `rag-stats --collections` and verify the active collection appears in the list
- Run inside Docker-like environment (simulated) with env-var overrides and verify config resolution works
