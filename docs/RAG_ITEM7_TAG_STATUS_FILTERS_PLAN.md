# Item 7 — Expose `--tag` / `--status` filters (query CLI + API)

**Status: COMPLETED (merged 2026-07-20, commit `e6b90dc`).** Everything described below as
"still to implement" is now live in `src/rag/query.py`, `src/rag/api/schemas.py`,
`src/rag/api/routes/query.py`, and `config*.yaml` (`tag_fetch_k`). Kept as the implementation
record for Roadmap item 7 (see `CLAUDE.md` → Known Limitations & Improvement Roadmap).

## Context

The vault carries a curated tag vocabulary and a `status` field per note — both strong
precision signals — but the retrieval layer never exposes them. `build_where` (`src/rag/query.py:30`)
only builds native `$eq` clauses for `domain`/`subdomain`/`type`/`source`/`confidence`, and
`search()` has no tag concept. This is Item 7 of the RAG quality roadmap (CLAUDE.md): let users
narrow retrieval by `status` (a scalar → native Chroma `$eq`) and by `tags` (stored as a
comma-joined string because Chroma 0.6.3 has weak array support → must be a post-filter, never a
`where` clause).

**Hard constraint (regression-critical):** when no tag/status filter is supplied, ranking must be
byte-for-byte unchanged. `build_where()` with all-None still returns `None`; `search()` with
`tags=None` must not widen the fetch pool or alter records. `make eval` must match baseline.

## Current state (important — work is partially started)

The branch `fix/query-tag-status-filters` already exists and is checked out (HEAD == `main`, no
commits ahead; changes are uncommitted in the working tree). `git diff -- src/rag/query.py` shows
these pieces are already done:

- `build_where(...)` has the `status=None` param + `{"status": {"$eq": status}}` clause and an
  updated docstring. ✓
- `search(...)` has the `tags=None` keyword param and a docstring describing the post-filter +
  `tag_fetch_k` widening. ✓ (signature/doc only)

Still to implement (the actual behavior + all downstream wiring + tests + config):

## Implementation

### 1. `src/rag/query.py` — `search()` body (the real behavior; only the signature/doc exist)

- **Fetch widening:** right after `fetch_k = max(n_results, rerank_fetch_k) if rerank else n_results`
  (`query.py:139`), add: `if tags: fetch_k = max(fetch_k, int(config.get("tag_fetch_k", 200)))`.
- **Tag post-filter:** after `records` is built (`query.py:154-157`) and before the
  `if rerank and records:` block (`query.py:162`), if `tags`:
  - Normalize the request: `want = {t.strip().lower() for t in tags if t and t.strip()}`.
  - Keep record `r` iff `want` is a subset of its tag set, where the record's tag set is
    `{s.strip().lower() for s in (r["metadata"].get("tags") or "").split(",") if s.strip()}`.
  - Use `(meta.get("tags") or "")` so `None`/missing/empty never raises → such records are
    excluded (they have no tags to match). Exact membership: `--tag ci` must NOT match `ci-cd`.
    Multiple tags = AND (subset test gives this for free).
  - Filter runs before rerank so the cross-encoder scores the filtered pool, then the existing
    rerank/else block trims to `n_results` unchanged.
- **Logging:** extend the existing `logger.info(...)` (`query.py:174`) to include `tags=%s` (pass
  `tags`) alongside `filter`/`rerank`.

### 2. `src/rag/query.py` — CLI `main()`

- Add args (near `query.py:196-205`): `--status` (single, `default=None`) and
  `--tag` (`action="append"`, `default=None`, repeatable; help notes exact match, AND semantics).
- Pass `status=args.status` into the `build_where(...)` call (`query.py:217`) and
  `tags=args.tag` into `search(...)` (`query.py:218`).
- Add two epilog examples (`query.py:183-190`), e.g.
  `rag-query "kubernetes" --tag devops -n 5` and `rag-query "deployment" --status processed`.
- Printed `Filter:` line (`query.py:230-231`): `status` already rides inside `where` via
  `build_where`, so it shows automatically. Tags are NOT in `where` → print them explicitly.
  Change the guard to fire on `where or args.tag` and append `tags=[...]` when `args.tag` is set
  (keep the existing `where` JSON when present).

### 3. `src/rag/api/schemas.py` — `QueryFilters` (`schemas.py:29`)

- Add `status: Optional[str] = None` and `tags: Optional[list[str]] = None`.
- Note: the module already uses builtin generics (`list[dict[str, Any]]` in `QueryResponse`,
  `schemas.py:56`), so use `Optional[list[str]]` — no new `List` import needed (deviates from the
  spec's "import `List`" only to match existing style; functionally identical).

### 4. `src/rag/api/routes/query.py` — `/query` handler (`routes/query.py:37-54`)

- Add `status=f.status if f else None` to the `build_where(...)` call.
- Add `tags=f.tags if f else None` to the `rag_query.search(...)` call.
- No other changes; `reranked` logic unaffected.

### 5. `config.yaml` — add `tag_fetch_k`

- Add `tag_fetch_k: 200` near `rerank_fetch_k` (`config.yaml:63`) with a one-line comment:
  the dense candidate pool floor when a `--tag` filter is active (tags are post-filtered, so the
  pool must be wide enough to keep matches).

## Tests (offline — `make test-unit`)

### `tests/test_query_search.py`

Current `FakeColl` (`test_query_search.py:8`) builds metadata as `{"title": d, "path": d}` — no
`tags`. Extend it (or add a sibling fake) so returned metadata can carry a per-doc `tags` string;
`FakeColl.last_n` already records the requested `n_results` for the widening assertion. Add:

- `build_where(status="processed")` → `{"status": {"$eq": "processed"}}`; combined with
  `domain=...` → `$and` of both. `build_where()` all-None → `None` (guard the no-op path).
- `search(tags=["devops"])` keeps only exact-tag-superset records (case-insensitive), drops
  non-matches and empty/missing-`tags` records.
- `--tag ci` does NOT match a record tagged `ci-cd` (exact membership).
- Two tags = AND (record must contain both).
- Widening fires: with `tags` set, `coll.last_n == 200` (default `tag_fetch_k`); with `tags=None`,
  `last_n` is unchanged from today (5 dense-only / 20 reranked) — proves no regression.

### `tests/test_api.py`

Reuse the `patch_search` / `SearchRecorder` pattern (`test_api.py:199-241`, which patches
`rag.query.search` while the REAL `build_where` runs). Add:

- `QueryFilters` accepts `tags` + `status` (POST body round-trips, 200).
- Route forwards them: `status` lands in `rec.last["filters"]` as a `$eq` clause via `build_where`
  (assert the where-dict like `test_query_filters_map_to_build_where`, `:297`); `tags` arrives as
  `rec.last["tags"]`.

### CLI (light)

Assert argparse accepts `--status` and repeated `--tag` (parse a small argv, check `args.status`
and `args.tag == [...]`) — no index needed.

## Verification

- `.venv/bin/python -m pytest tests/` (`make test-unit`) green, new cases included. (Install dev
  deps first if needed: `uv pip install -e ".[dev]"`.)
- `make eval` unchanged vs `tests/eval/baseline.json` — golden queries are unfiltered by design, so
  this confirms the no-filter path did not regress ranking.
- Optional manual smoke (needs a populated index): `.venv/bin/rag-query "kubernetes" --tag devops -n 5`
  and `.venv/bin/rag-query "deployment" --status processed`.
- `git diff main...HEAD --stat` and summarize. Do NOT open a PR — stop and report.

## Guardrails

- `.venv` only — never bare `python`/`python3` (Conda base leaks in).
- Reuse `build_where`/`search` — do not duplicate retrieval. Tags are a post-filter, never a fake
  `where` clause. `status` is native `$eq` in `build_where`.
- The repo working tree has mode-bit/whitespace noise across many files (git status shows ~28 files
  modified with 0 line changes). Stage explicit paths only — never `git add -A`. Files to stage:
  `src/rag/query.py`, `src/rag/api/schemas.py`, `src/rag/api/routes/query.py`, `config.yaml`,
  `tests/test_query_search.py`, `tests/test_api.py`.
- Commit with a clear message; no PR.
