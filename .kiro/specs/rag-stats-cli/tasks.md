# Implementation Plan

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - No CLI Command for Index Statistics
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate no `rag-stats` command exists and users cannot inspect index statistics without internal knowledge
  - **Scoped PBT Approach**: Scope the property to concrete failing cases: invoking `rag-stats` as a console script entry point, and verifying the stats module exists with a working `main()` function
  - Test that `rag.stats` module does not exist (ImportError) on unfixed code
  - Test that `rag-stats` is not registered in `pyproject.toml` `[project.scripts]`
  - Test that calling a hypothetical `stats.main()` with a mocked store produces correct output (total count, per-source-type breakdown via paginated `iter_records`, collection listing via `list_collection_names`)
  - Property: for any store containing N chunks across S source types, `main()` SHALL print the correct total (matching `store.count()`), correct per-type counts summing to N, and correct unique file counts per type
  - Run test on UNFIXED code - expect FAILURE (ImportError for `rag.stats` confirms the bug exists)
  - Document counterexamples found (e.g., "from rag.stats import main raises ImportError")
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4_

- [x] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Existing Store Interfaces and Import Invariant Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe: `get_store(config)` returns a `ChromaStore` with `.count()`, `.iter_records()`, `.name` on unfixed code
  - Observe: `list_collection_names(config)` returns a set of collection name strings on unfixed code
  - Observe: `drop_collection(config, name)` returns an int (pre-drop count) on unfixed code
  - Observe: `chromadb` is only imported in `src/rag/store/chroma_store.py` (no other file under `src/rag/`)
  - Write property-based test: for all valid config dicts (with `store: "chroma"` and a temp `index_path`), `get_store` returns a `ChromaStore` instance with the expected `name` property matching the collection_name
  - Write property-based test: for all valid config dicts, `list_collection_names` returns a set (type-stable interface)
  - Write property-based test: scanning all `.py` files under `src/rag/` (excluding `src/rag/store/chroma_store.py`), none contain `import chromadb` or `chromadb.` — this must hold after the new `stats.py` is added
  - Verify all preservation tests pass on UNFIXED code
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 3. Implement `rag-stats` CLI command

  - [x] 3.1 Create `src/rag/stats.py` module
    - Add `argparse` with `--collections` flag and `--page-size` (default 10,000) argument
    - Import `load_config` from `.utils` and `setup_logging` from `.utils`
    - Import `get_store` and `list_collection_names` from `.store`
    - Implement paginated aggregation: iterate `store.iter_records(page_size)`, accumulate `type_counts` dict and `type_files` dict (set of unique paths per source_type)
    - Print `Collection: {store.name}`, `Total chunks: {total:,}`, and per-source-type breakdown
    - When `--collections` is passed: list all collection names via `list_collection_names(config)` and return early
    - Use Python 3.10 syntax only (no `match`, no `type X = ...`, no `ExceptionGroup`)
    - Do NOT import `chromadb` anywhere in this file — use only the store abstraction
    - _Bug_Condition: isBugCondition(input) where input.intent in ['check_chunk_count', 'check_source_breakdown', 'list_collections'] AND NOT existsCLICommand(input.intent)_
    - _Expected_Behavior: rag-stats prints accurate total count (from store.count()), per-source-type chunk counts summing to total, unique file counts, all via paginated iteration_
    - _Preservation: No modification to existing store interfaces; no chromadb import outside src/rag/store/_
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.4_

  - [x] 3.2 Register `rag-stats` console script in `pyproject.toml`
    - Add `rag-stats = "rag.stats:main"` to `[project.scripts]` section
    - Re-install the package in editable mode: `uv pip install -e . --no-deps`
    - _Requirements: 2.3_

  - [x] 3.3 Add Makefile targets `stats` and `jetson-stats`
    - Add `make stats` target: runs `.venv/bin/rag-stats $(ARGS)`
    - Add `make jetson-stats` target: runs via `docker compose -f docker-compose.jetson.yml run --rm rag rag-stats $(ARGS)`
    - Add both to `.PHONY` declaration
    - Add help text entries in the `help` target
    - _Requirements: 2.3_

  - [x] 3.4 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Stats Command Produces Accurate Output
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (correct total, correct breakdown, correct file counts, paginated iteration)
    - When this test passes, it confirms the expected behavior is satisfied
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.5 Verify preservation tests still pass
    - **Property 2: Preservation** - Existing Store Interfaces and Import Invariant Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm store interfaces unchanged, chromadb import confined, existing commands unaffected
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 4. Checkpoint - Ensure all tests pass
  - Run `make test-unit` to confirm the full offline test suite passes (including the new stats tests)
  - Verify `test_chromadb_import_confined_to_chroma_store_module` still passes (no chromadb import in `src/rag/stats.py`)
  - Verify `rag-stats --help` prints usage without errors
  - Verify `rag-stats --collections` lists collections without errors (against a real or test index)
  - Ensure all tests pass, ask the user if questions arise.
