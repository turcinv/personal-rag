# Synthetic stale-trap fixtures for `make eval`

Plan only. No code written, no index touched, no reindex run. Companion to
`.specs/keeping-it-true.md` §3, which implemented the stale-trap *scoring*
(`stale_trap_passed`, the `stale_trap` block, the `kind`/`must_not` JSONL
fields) but deliberately shipped **zero** stale-trap rows — Vít's content rule
is "don't invent vault content," and §3 could not confirm real note pairs
without his sign-off. He has now signed off on a different mechanism: synthetic
fixture pages, not real notes (decision below). This spec is the content +
harness plan for that mechanism.

Each claim is tagged **[Observed]** (file+line / command output), **[Inferred]**
(reasoned from Observed facts), or **[Proposal]** (what to build). A "not
read / not verified" list is at the end.

---

## 0. Branch state — read before anything else lands

- **[Observed]** `~/Documents/personal-rag-keeping-it-true` is currently on
  `fix/drift-py310-utc` (`git branch --show-current`), which is **already
  merged** into `main` via PR #13 (`git log --oneline --graph --all`: `d84c865
  Merge pull request #13 from turcinv/fix/drift-py310-utc`, parent of
  `09f2513 Merge pull request #12 from turcinv/feature/keeping-it-true-2`).
  `feature/keeping-it-true` (588495d) and `feature/keeping-it-true-2`
  (4f656e0) are also both already merged into `main` by that same merge chain.
- **[Observed]** `main` locally is `6a2f2ed`, tracking
  `origin/main: behind 17` — the merges above exist on `main` upstream but the
  local `main` ref in this worktree has not been fast-forwarded.
- **[Inferred]** None of `fix/drift-py310-utc`, `feature/keeping-it-true`,
  `feature/keeping-it-true-2` is a safe base to branch from for *new* work —
  they are closed/merged lines, and starting on any of them would mix this
  feature's history with already-landed, already-reviewed commits (the exact
  "landing on someone else's WIP" failure mode `impl-planner`'s hard rules
  warn about, generalized to "landing on a dead branch").
- **[Proposal]** Before the coding agent touches anything: `git fetch origin`,
  then branch `feature/stale-trap-fixtures` from `origin/main` (**not** from
  the current `fix/drift-py310-utc` checkout state), e.g.:
  ```
  git fetch origin
  git checkout -b feature/stale-trap-fixtures origin/main
  ```
  This is a **writes/shell-run** step and is not pre-approved by this plan
  (impl-planner does not run it) — the coding agent creates it, after Vít
  approves this spec, as the first step of implementation.

## 1. Decision this spec implements (Vít, 2026-10-08)

> The trap is built from synthetic fixture pages, not from real stale vault
> notes. The three real stale notes found in the vault were corrected the same
> day, so a trap built on them would pass forever and test nothing.

This **supersedes** the "golden rows — NOT added" section of
`.specs/keeping-it-true.md` §3, which had proposed picking real
`needs-review`/superseded vault notes as trap candidates. That proposal is
rejected by this decision, not merely left undone.

## 2. What already exists (do not re-implement)

- **[Observed]** `src/rag/eval.py:62-73` — `stale_trap_passed(records, expected,
  must_not)`: passes iff the first `expected` note ranks and no `must_not` note
  ranks above or at it.
- **[Observed]** `src/rag/eval.py:106-139` — `evaluate()` routes `kind ==
  "stale-trap"` rows into a separate `stale_trap` dict
  (`{"passed", "total", "rows"}`), excluded from `overall`/`by_kind`
  aggregation; every other row is scored exactly as before (byte-identical,
  since `must_not`/`kind` default to absent).
- **[Observed]** `src/rag/eval.py:179-196` — `print_report()` prints a
  `STALE-TRAP n/total passed` block with per-row PASS/FAIL + ranks when
  `stale_trap.total > 0`; prints nothing when `total == 0` (today's state).
- **[Observed]** `tests/test_eval.py` already covers `stale_trap_passed` pure
  logic and `evaluate()` routing with synthetic in-memory record lists (no
  store, no model) — these tests pass today and are **not** touched by this
  plan.
- **[Observed]** `tests/eval/golden_queries.jsonl` has 45 rows, all
  `kind: "vault"|"resource"`, none `"stale-trap"`
  (`wc -l` → 45; `grep` confirms no `stale-trap` line exists).
- **[Observed]** `tests/test_eval.py::test_golden_set_wellformed` already
  accepts `"stale-trap"` as a valid `kind` and validates `must_not` is a list
  when present (`item["kind"] in {"vault", "resource", "stale-trap"}`) — so
  adding real stale-trap rows to the golden file requires **no** test-schema
  change, only new JSONL lines once the fixture mechanism below produces real
  rank data for them.

**What is missing and in scope here:** (a) synthetic fixture *content* for the
three supersession topics named in the brief, (b) an isolated ingest/store
mechanism that gets those fixture pages through the real chunking/embedding/
`search()` path without touching the vault or the production Chroma
collection, (c) a `make eval`-reachable way to run the stale-trap rows against
that isolated store while still running the 45 real golden rows against the
real production index in the same invocation (per the acceptance shape), and
(d) teeth tests proving the scoring isn't vacuous.

## 3. Isolation mechanism — decision + justification

### 3.1 The constraint, precisely

**[Observed]** `evaluate(golden, *, n=10, config=None, collection_name=None,
rerank=False, hybrid=False)` (`src/rag/eval.py:106`) opens exactly **one**
store for the **whole** golden list via `open_store(config, collection_name,
model=model)` (`eval.py:112`) and runs every row — vault, resource, and
stale-trap alike — through that one `store`. There is no per-row store
switch today.

**[Observed]** `search()` accepts an already-open `store=` kwarg
(`src/rag/query.py:118`, `store=None` default, resolved via `open_store` only
if not given) — so the seam already supports "caller supplies the store,"
which is exactly the hook synthetic fixtures need, with **zero** change to
`search()` itself.

**[Observed]** `open_store()` calls `ensure_compatible(store,
expected_provenance(...))` (`query.py:106-110`), which aborts on a fingerprint
mismatch (embedding model/revision/dimension/chunker version/chunk
sizes/metric/corpus_profile — `.specs/keeping-it-true.md` §0, confirmed against
`src/rag/provenance.py:25`). A fixture store built with the **same**
embedding model and chunk settings as the production profile satisfies this
automatically; a store built with a fake/mocked model (as `test_end_to_end.py`
does) would **not** be provenance-compatible with the real model and must
skip `ensure_compatible` for its own internal queries — acceptable, since the
fixture store is only ever opened by fixture code, never by `open_store()`
against the real config.

### 3.2 Options considered

**(a) A separate Chroma collection in the *same* `index_path` directory**
(e.g. `collection_name="stale_trap_fixtures"` inside `./chroma_db`).
Rejected: it still physically lives inside the production SQLite file
(`chroma_db/chroma.sqlite3`), so a backup/restore, a `rag-stats` sweep, or a
future "list collections" admin action over that `index_path` would surface
synthetic rows unless every such tool is taught to filter them out. It also
means `make backup`/`make backup-verify` capture fixture bytes alongside real
data forever. Does not meet "must not enter the production Chroma collection"
in spirit even though it's technically a different *collection* — the brief's
wording and the risk (admin tooling, Jetson sync of `chroma_db/`) both point
at directory-level isolation, not collection-name isolation.

**(b) A temporary store built at eval time, in a `tempfile.mkdtemp()` /
`tmp_path`-style directory, torn down after the stale-trap rows are scored.**
This is the `test_end_to_end.py` pattern (`ChromaStore(index_path, name)` over
`tmp_path / "chroma"`, `src/rag/store/chroma_store.py`'s `ChromaStore`
constructor takes a bare path+name, no global state) applied to a `make
eval`-reachable script instead of a pytest fixture. **Chosen.** Justification:
- **[Observed]** `ChromaStore.__init__` takes `(index_path, collection_name)`
  and nothing else is global/shared (confirmed by `test_end_to_end.py` and
  `test_store.py` both instantiating independent `ChromaStore`s over different
  `tmp_path`s in the same process without interference).
  **[Not verified]**: I have not opened `src/rag/store/chroma_store.py` body
  (only its import surface via `store/__init__.py`) to confirm there is no
  module-level client cache keyed by something coarser than
  `(index_path, name)`; flagged below as a pre-implementation check, not
  assumed.
- Physically never under `vault_path` or the production `index_path`
  (`./chroma_db`), so it cannot be swept into `make backup`, `rag-stats`, or
  the Jetson sync of the real store, and a crash mid-run leaves no residue
  beyond an OS temp directory.
- Reuses the **real** `extract_md_file` (chunking), the **real**
  `index_file_chunks`/`embed_and_upsert` (ingest), the **real** embedding
  model (same `embedding_model` as `config.yaml`, loaded once and shared — see
  §4), and the **real** `search()` — satisfying "keep the ingest and search
  code paths the same as production" exactly, with no fake/mock model (unlike
  `test_end_to_end.py`'s `_FakeModel`, which is right for a fast pytest unit
  test but wrong here: a fake dim-4 model can't exercise real semantic
  ranking, which is the entire point of a ranking trap).
- Costs one extra model-encode call for ~6-12 small chunks per `make eval`
  run (3 topics × 2 pages × ~1-2 chunks each) — negligible next to embedding
  the 45 golden-set queries themselves, which already load the same cached
  model (`get_model`, `query.py:76`, process-local cache).

**(c) In-memory fake store (no real Chroma, hand-rolled nearest-neighbor over
Python lists).** Rejected: would not exercise "the default search() path"
faithfully — `search()` calls `store.query(...)` expecting the
`RetrievalStore` Protocol's real distance semantics, filtering, and the
`needs-review` post-filter operates on `metadata["status"]` the same way
regardless of backend, but building a second store implementation purely for
this fixture duplicates `ChromaStore` behavior and risks silently drifting
from it. `ChromaStore` over a temp dir is no slower and has zero duplication
risk.

### 3.3 Why this doesn't touch production

- The fixture store's `index_path` is a fresh temp directory created and
  destroyed per `make eval` invocation (or per pytest run for the teeth
  tests) — never `./chroma_db`, never anything under `vault_path`.
- The fixture pages are never written as files under `vault_path` — they are
  constructed in-memory as `(path, title, status, body)` tuples and passed
  directly to `extract_md_file`-equivalent logic (see §4.2: `extract_md_file`
  takes a `Path` and reads from disk, so the harness writes the synthetic
  `.md` files to a **temp vault directory**, not the real vault — still
  isolated, just needs a temp source dir in addition to a temp index dir).
- `rag-drift` and `rag-index` are never invoked by the fixture harness; it
  calls the ingest primitives (`extract_md_file`, `index_file_chunks`)
  directly, the same functions `rag-index`/`rag-drift` call internally, so the
  acceptance criterion "`rag-drift` stays clean after `make eval`" holds
  trivially — nothing the fixture harness does is visible to a `rag-drift`
  run against the real `vault_path`/`index_path`, because neither is touched.

## 4. Fixture content

### 4.1 Source facts

| Trap | Stale claim | Current claim | Source note (vault title) |
|---|---|---|---|
| T1 | async submit then poll `GET /api/check/{id}` | `GET /api/check/{id}` and the in-memory `internal/job` store were removed (parser-generator#3, MR !12, merged 2026-07-28); `POST /api/submit` and `POST /api/generate/{id}` run the pipeline inline and return 200; `{id}` remains a UUID-validated correlation token echoed in error bodies | "Logmanager Parser Generator Sync Refactor" |
| T2 | RunPod serverless production | AWS EKS `lm-eks`, g5.xlarge GPU node | "Logmanager Alert and Parser Generator EKS Deployment on AWS" |
| T3 | `transformers_cfg` + `LogitsProcessor` grammar enforcement | vLLM + xgrammar | "Bedrock Structured Outputs vs the Alert Generator EBNF Grammar" |

**[Observed, verified against the vault 2026-10-08 by Vít]** T1's "current"
fact is corrected from the original brief: the earlier draft said current
POST endpoints return 200 "no polling, no job id" — **wrong**. The vault note
"Logmanager Parser Generator Sync Refactor" states `{id}` is **not** removed:
it remains as a UUID-validated correlation token echoed in error bodies, even
though polling (`GET /api/check/{id}`) and the in-memory job store are gone.
The T1 current-page wording in §4.2 below reflects this: "no polling; `{id}`
stays as a correlation token" — never "no job id". T2 and T3's facts as
stated above match the vault as written; no outstanding correction for them.

### 4.2 Fixture page shape

Each topic gets **two** synthetic pages, both frontmatter + body, written as
real `.md` files into a temp "fixture vault" directory at eval/test-run time
(not committed as static files — generated by the harness so they can share
one `SYNTHETIC: true` marker constant and stay trivially auditable; see §4.4
for the rationale on generating vs. committing them statically):

```markdown
---
title: "<Synthetic> Parser Generator Job Status Endpoint (stale)"
status: needs-review   # or: processed  -- see the two trap variants below
synthetic: true
type: fixture
---

**SYNTHETIC FIXTURE — stale-trap test page, not a real vault note.**

Historically, the parser-generator submitted a parse job asynchronously and
polled `GET /api/check/{id}` for completion status, using an in-memory job
store keyed by job id.
```

```markdown
---
title: "<Synthetic> Parser Generator Inline Response (current)"
status: processed
synthetic: true
type: fixture
---

**SYNTHETIC FIXTURE — stale-trap test page, not a real vault note.**

As of parser-generator#3 (MR !12, merged 2026-07-28), `GET /api/check/{id}`
and the in-memory job store were removed. `POST /api/submit` and
`POST /api/generate/{id}` run the pipeline inline and return 200: no polling;
`{id}` stays as a correlation token (a UUID, validated and echoed back in
error bodies, but no longer used to poll for status).
```

**[Observed, per the correction above]** This wording deliberately keeps
`{id}` in the current page's text, as a correlation token, rather than
implying it disappeared — the earlier draft's "no job id" was the factual
error Vít flagged and this fixes.

- **[Proposal]** `title` is prefixed `"<Synthetic> "` (visible even if a
  fixture page ever leaked into a real listing) and the body opens with a
  bold **SYNTHETIC FIXTURE** line — satisfies "mark fixture pages clearly as
  synthetic in frontmatter and body" with two independent, redundant markers.
- **[Proposal]** `synthetic: true` frontmatter key, plus `type: fixture` — the
  latter reuses the existing `type` metadata field (`extract_md_file` already
  reads `meta.get("type")`, `markdown.py:56`) so a fixture chunk is
  trivially excludable by any future tooling that filters on `type`, with no
  new metadata plumbing.
- **[Proposal]** Each page is short enough to stay a **single chunk** after
  `split_by_headings`/`chunk_text` (no heading inside the body, well under
  `chunk_max_chars` default 1200) — keeps rank assertions in the teeth tests
  unambiguous (one chunk per page, no intra-page rank competition).
- Content is paraphrased from the facts in §4.1 (T1 corrected and verified
  against the vault by Vít; T2/T3 confirmed by Vít to match the vault as
  written — §7), not copied verbatim from the real vault notes' prose.

### 4.3 The three trap pairs, with status assignment (meets the ≥1-needs-review
+ ≥1-processed requirement)

| # | Topic | Stale page `status` | Current page `status` | Tests | Expected outcome |
|---|---|---|---|---|---|
| T1 | parser-generator async job endpoint | `needs-review` | `processed` | default exclusion of the stale page (needs-review filter must drop it from the pool entirely); deterministic precondition (§6.3) | **passes** |
| T2 | alert-generator production infra | `processed` | `processed` | documented baseline only — no filter defends it; query phrased in the stale page's vocabulary | **xfail (strict)** — see §4.5 |
| T3 | grammar enforcement library | `needs-review` | `processed` | default exclusion again, second instance for redundancy/robustness across topics; deterministic precondition (§6.3) | **passes** |

- **[Proposal]** T1 and T3 cover "stale page has `status: needs-review`"
  (brief requires **at least one**; two given for robustness — if the
  `needs-review` post-filter's pool-widening (`tag_fetch_k`, `query.py:138`)
  interacts oddly with only a single-chunk corpus on T1, T3 is a second,
  independent check of the same mechanism).
- **[Proposal, revised per Vít 2026-10-08]** T2 covers "stale page has
  `status: processed`" (brief requires **at least one**), but is **no longer
  expected to pass**. §4.5 explains why: nothing in `search()` prefers a newer
  of two equally-`processed` notes on the same topic, so a passing T2 would
  only reflect how the fixture happened to be worded, not a real defense. T2
  is kept as a documented `xfail` baseline instead of a third passing row.
- This gives **3 rows total** (2 passing + 1 documented xfail), inside the
  brief's "2-3 rows" window, each pair independently sourced from a different
  real supersession so a single bad fixture pair doesn't collapse the whole
  stale-trap block to 0 signal.

### 4.5 T2 as a documented xfail baseline — rationale (Vít, 2026-10-08)

**[Decision]** T2 (RunPod serverless vs. EKS/g5.xlarge, both `status:
processed`) is **not** a trap personal-rag is expected to pass today.
`default_excluded_status` (§6's precondition work confirms
`config.yaml:81` sets `["needs-review"]`) only ever excludes `needs-review`
chunks; it has no concept of "superseded" vs. "superseding" among two
`processed` notes on the same topic, and `search()` has no recency- or
freshness-aware re-ranking anywhere in its pipeline (confirmed by
`query.py`'s full body, read in full for this plan — the only post-filters
are `tags` and `default_excluded_status`; there is no date/freshness signal).
A T2 fixture that happened to pass would therefore only be reporting "MiniLM's
embedding distance for these two specific paragraphs favors the current one"
— an accident of wording, not a property of the system. Treating that as a
passing acceptance row would be the exact failure mode the brief's own
preamble warns about ("a trap built on them would pass forever and test
nothing"), just relocated from real notes to a fixture.

Keeping T2 in the suite **as a deliberately-failing baseline** instead of
dropping it serves a real purpose: it is the proof that, absent
`needs-review`, nothing defends against a stale-but-`processed` note, which
is exactly the gap `.specs/keeping-it-true.md` §2's design (status-based
exclusion only) accepts. Marking it `xfail(strict=True)` means if a future
change (e.g. a freshness signal, `.specs/keeping-it-true.md`'s roadmap
mentioning none exists today) ever makes T2 start passing, the strict xfail
**fails the test suite** — forcing a human to notice and decide whether that
new passing behavior is real retrieval improvement or another wording
accident, rather than letting it slide in silently.

### 4.4 Generated-at-runtime vs. committed static files — decision

**[Proposal]** The three topics' fixture page *content* (the markdown
templates in §4.2, parameterized only by which `status` goes on which page per
§4.3) are defined as **Python string constants** in a new module
`src/rag/eval_fixtures.py`, not as static `.md` files committed under
`tests/eval/fixtures/` or similar. Rationale:
- They are never read by a human browsing the repo as "vault content" (there
  is no vault-shaped directory to accidentally sync/mirror) — directly
  supports "synthetic pages must not go into the vault."
  A static `.md` file sitting in the repo risks a future `rsync`/`cp -r`-style
  sync script (the brief's own context flags the Jetson vault sync as a sharp
  edge — `AGENTS.md` "Syncing the index to the Jetson") treating any `.md`
  tree as vault-shaped and copying it; a Python constant cannot be mistaken
  for a vault note by any path-glob-based tool.
- One module is trivially greppable/auditable (`grep synthetic
  src/rag/eval_fixtures.py` shows all three topics at once) and trivially
  diffable in review, vs. six small scattered files.
- The harness (§5) writes these constants out to real `.md` files inside a
  **temp directory it creates and destroys per run** — so `extract_md_file`
  still gets a real `Path` on real bytes on disk, preserving "keep the ingest
  ... code paths the same as production" (no shortcut that skips file I/O).

## 5. Harness design

### 5.1 New module: `src/rag/eval_fixtures.py`

**[Proposal]** Responsibilities:
1. `FIXTURE_PAGES: list[dict]` — the 6 synthetic page definitions (3 topics ×
   2 pages), each `{"rel_path": ..., "title": ..., "status": ..., "body": ...,
   "topic": ...}`. `rel_path` values live under a clearly-synthetic subpath,
   e.g. `"Synthetic/Stale-Trap/<topic-slug>-stale.md"` /
   `"...-current.md"` — chosen so that even if someone mis-pointed `vault_path`
   at the temp dir by mistake, the path itself signals non-vault content.
2. `STALE_TRAP_GOLDEN: list[dict]` — the 3 JSONL-shaped rows per §3's exact
   schema (`query`, `expected`, `must_not`, `kind: "stale-trap"`), referencing
   the fixture pages' `title` strings as substrings (matching `is_hit`'s
   title-or-path substring semantics, `eval.py:40`), **plus one additional
   key not in the committed-JSONL schema**: `baseline: bool` (default
   `False`), set `True` only on the T2 row. T1/T3 queries are phrased in the
   **stale page's own vocabulary** (§6.3 — e.g. T1: "How does the frontend
   poll for parser job status?"), not the current page's, so the trap
   actually has to overcome a lexically-stale-favoring query. T2's query is
   likewise phrased in the stale page's vocabulary (RunPod serverless
   wording, per §4.5) — it is expected to find the stale page and is kept
   that way deliberately, not softened to make it pass.
3. `build_fixture_store(config, tmp_dir) -> store` — a function that:
   - writes `FIXTURE_PAGES` out as real `.md` files (with YAML frontmatter)
     under `tmp_dir / "vault"`;
   - opens a fresh `ChromaStore(tmp_dir / "chroma", "stale_trap_fixtures")`
     (no `ensure_compatible` call against this store — it is never opened via
     `open_store()`/the production config, so no provenance fingerprint check
     applies or is needed);
   - for each page, calls `extract_md_file(path, tmp_dir / "vault", config,
     max_chars, overlap)` — the **real, unmodified** extractor — then
     `index_file_chunks(ids, docs, metas, reconciliation, model, device,
     embed_batch, store)` with a **fresh, empty**
     `ReconciliationCatalog` (so every chunk is classified `new`, i.e. always
     embedded — this is a one-shot build-and-discard store, not an
     incrementally-maintained one) and the **same real embedding model**
     `evaluate()` already loaded for the golden set (passed in, not reloaded —
     one model load per `make eval` run, same as today);
   - returns the populated `store`.
4. `run_stale_trap(config, model, n=10, rerank=False, hybrid=False) ->
   list[dict]` — opens a `tempfile.TemporaryDirectory()`, builds the fixture
   store via (3), runs each `STALE_TRAP_GOLDEN` row through `search(...,
   store=fixture_store, config=config, model=model, rerank=rerank,
   hybrid=hybrid)` (passing the already-open fixture store so `search()`
   never calls `open_store()`/`ensure_compatible` against it), scores each
   with `stale_trap_passed`, and returns the same per-row dict shape
   `evaluate()` already builds at `eval.py:120-127` — then the temp directory
   (and the fixture store with it) is deleted on context exit, before
   `run_stale_trap` returns.

### 5.2 Changes to `src/rag/eval.py`

**[Proposal]** `evaluate()` (`eval.py:106`) gains **no new required
parameter** — golden-set stale-trap rows (`kind == "stale-trap"` present in
the `golden` list passed in) continue to be scored exactly as today (§2),
unchanged. What's new: `evaluate()` additionally calls
`eval_fixtures.run_stale_trap(config, model, n=n, rerank=rerank,
hybrid=hybrid)` once per invocation and **merges** its rows into the
`stale_trap` block alongside any `kind: "stale-trap"` rows that happen to be
in the golden JSONL (there are none today — §2 — so in practice the merged
block is 100% fixture-sourced rows until/unless someone later adds JSONL
stale-trap rows too). This keeps the mechanism general: the harness and the
golden-JSONL path both feed the same `stale_trap` aggregation, so a future
real-note trap (if one is ever found that doesn't "pass forever," per the
brief's own caveat) could still be added as a JSONL row without a schema
change.
- **[Proposal, revised per Vít 2026-10-08]** Rows carrying `baseline: True`
  (T2 only, §4.5) are **excluded from `stale_trap["passed"]`/`["total"]`**
  and reported in a **separate** list, e.g.
  `stale_trap["baseline_rows"]` (each still carrying `expected_rank`,
  `must_not_rank`, and `passed` for visibility — `passed` is expected
  `False` here, not omitted, since the row shape stays uniform for
  `print_report()`). `print_report()` (`eval.py:168`) gets a second small
  block under the `STALE-TRAP n/total passed` line:
  `BASELINE (expected fail, not counted)` listing each baseline row's
  PASS/FAIL mark — a baseline row showing **FAIL** is the healthy, expected
  state; **[Proposal]** if a baseline row ever shows PASS in a `make eval`
  run, `print_report()` prints a one-line `⚠ baseline now passing — see
  .specs/stale-trap-fixtures.md §4.5` warning, mirroring the strict-xfail
  intent in the pytest layer (§6) for the human-facing `make eval` output,
  since `make eval` itself has no pass/fail exit code to fail on this the
  way pytest's strict xfail does.
- **[Proposal]** Add an `include_stale_trap_fixtures: bool = True` keyword to
  `evaluate()` defaulting **True**, so `make eval` runs it by default (meets
  "reports the stale_trap block with the new rows" without a new flag), but
  the teeth tests (§6) can pass `False` to isolate "golden-JSONL stale-trap
  scoring" from "fixture-harness stale-trap scoring" when asserting on one
  without the other.
- `evaluate()`'s doc comment at `eval.py:106` gets one line added noting the
  fixture rows are synthetic and isolated, and that baseline rows are
  excluded from the pass count by design (see §5.4 on doc-comment scope).

### 5.3 `make eval` / CLI surface

- **[Proposal]** No new Makefile target and no new CLI flag are required for
  the default `make eval` path — `include_stale_trap_fixtures` defaults True
  inside `evaluate()`, and `rag.eval.main()` (`eval.py:200`) already calls
  `evaluate(golden, n=args.n, config=config, collection_name=args.collection,
  rerank=rerank, hybrid=args.hybrid)` with no changes needed to pick up the
  new default-True behavior.
- **[Proposal]** Add `--no-stale-trap-fixtures` (CLI flag → `main()` passes
  `include_stale_trap_fixtures=False`) as an escape hatch for a fast
  iteration loop that only wants real-golden-set recall numbers and doesn't
  want to pay the extra model-encode + temp-store-build cost each run — a
  convenience, not required by the acceptance shape, but cheap and avoids
  anyone being tempted to comment out the fixture call by hand later.

### 5.4 Config

- **[Proposal]** No `config.yaml` change. The fixture harness reads
  `config["embedding_model"]`, `config["chunk_max_chars"]`,
  `config["chunk_overlap_chars"]`, `config["embedding_batch_size"]` from the
  **same** config `evaluate()` already has in hand — it does not need its own
  config file or profile. This directly satisfies "keep the 45 existing
  golden rows and the main recall/MRR numbers unaffected": nothing about
  `config.yaml`, `vault_path`, or `index_path` changes, so the real golden-set
  query path (`open_store(config, ...)` at `eval.py:112`, pointed at the real
  `./chroma_db`) is untouched.

## 6. Tests — the teeth requirement

**[Proposal]** New file `tests/test_eval_fixtures.py` (offline, no network, no
real vault, no real model-download risk beyond what `make test-unit` already
accepts — **flag below**: this is the one place this plan's tests load the
**real** `sentence-transformers/all-MiniLM-L6-v2` model rather than a fake
one, which is a deliberate deviation from `test_eval.py`'s pure-logic style;
see §6.4 for why and the fallback if that's unacceptable for `make test-unit`'s
offline guarantee).

### 6.1 Structural tests (no model, fast)
- `FIXTURE_PAGES` has exactly 6 entries, 3 topics; each topic has exactly one
  `needs-review`-or-`processed` stale page and one `processed` current page
  (per §4.3's table); every page's `title` starts with the synthetic marker
  and every body contains the `SYNTHETIC FIXTURE` marker string.
- `STALE_TRAP_GOLDEN` has exactly 3 rows, each `kind: "stale-trap"`, each
  `expected`/`must_not` a non-empty list of substrings that actually appear in
  some `FIXTURE_PAGES` entry's `title` (a static cross-check, catching a typo
  between the two constants without running any retrieval).

### 6.2 End-to-end harness test (real model, real ChromaStore, temp dirs)
- `test_run_stale_trap_t1_t3_pass_on_current_code()`: calls
  `run_stale_trap(config, model)` with the real small `config.yaml`-equivalent
  settings (model loaded once, module-scoped pytest fixture to amortize load
  cost across this file's tests) and asserts the **T1 and T3** rows'
  `passed == True`, and that `stale_trap["total"]` (passing-eligible rows)
  is 2, not 3 — T2 is excluded from this count by design (§4.5, §5.2) and is
  asserted separately as an xfail (§6.3). This is the acceptance shape's
  "T1 and T3 passing," run as a pytest rather than a full `make eval`
  invocation (the plan's "acceptance shape" also includes running the real
  `make eval` manually once post-merge, see §8).

### 6.3 Teeth tests — prove the scoring isn't vacuous (brief's explicit
requirement, revised per Vít 2026-10-08 for determinism)

The original draft of this plan scored T1/T3's "ranking" using whatever
query happened to be written and trusted that the real model would rank the
current page first "well enough" for the stale-trap to be meaningful. Vít
flagged this as embedding luck, not a guarantee — the query wording wasn't
chosen to make the stale page *competitive*, so a pass could be trivial (the
stale page was never a real contender) rather than proof the needs-review
filter is doing the work. The revised design below makes each T1/T3 test
assert an explicit **precondition** first, so the test is self-documenting
about what it actually proves.

**Precondition-driven design for T1 and T3** (`needs-review` stale pages):
1. Phrase each query in the **stale page's own vocabulary** — the words a
   person would use if they still believed the stale fact, not the words
   that describe the correction. E.g.:
   - T1: `"How does the frontend poll for parser job status?"` — "poll",
     "job status" are the stale page's terms (`GET /api/check/{id}`,
     in-memory job store); the current page's text ("inline", "correlation
     token", "no polling") shares none of that vocabulary.
   - T3: a query built around `transformers_cfg` / `LogitsProcessor`
     terminology (the stale page's library names), not `vLLM`/`xgrammar`
     (the current page's).
2. **Precondition assertion** (new, explicit): with
   `include_unreviewed=True` (the existing opt-in,
   `.specs/keeping-it-true.md` §2, `query.py:118`) against the fixture store
   for that topic's two-page corpus, assert the **stale page ranks #1**
   (`first_hit_rank(records, must_not) == 1`) — i.e. prove the query really
   does favor the stale page's wording when nothing filters it out. If this
   precondition does not hold with the real model, **the fixture wording is
   wrong, not the test** — per Vít's instruction, sharpen the query/page
   text until the precondition holds; never weaken or drop the assertion to
   make the test pass anyway. This is the mechanism that converts "ranking
   luck" into "deliberately engineered, verified-adversarial fixture."
3. **Main assertions**, both against the **same** fixture store/query, now
   that step 2 has proven the trap is adversarial:
   - with the **default** filter active (`include_unreviewed` not passed /
     `False`, i.e. ordinary `search()` call with
     `config["default_excluded_status"] == ["needs-review"]` as confirmed in
     `config.yaml:81` — §7's open item on this is now resolved, not merely
     assumed), `stale_trap_passed(records, expected, must_not)` is **True**;
   - with `include_unreviewed=True` (filter disabled), re-run the same query
     and assert `stale_trap_passed(...)` is **False** — this is only a
     meaningful assertion *because* step 2 already proved the stale page
     would otherwise win.

This yields two tests, each with three assertions (precondition, filter-on
pass, filter-off fail):
`test_t1_needs_review_trap_has_teeth()`,
`test_t3_needs_review_trap_has_teeth()`.

**T2 — dropped as a ranking-inversion test, replaced by an xfail baseline**
(per §4.5's rationale): the earlier draft's
`test_processed_row_fails_when_ranking_inverted()` (hand-reversing ranks or
monkeypatching `records.reverse()`) is **removed from this plan**. Per
Vít's instruction, that test only re-exercises `stale_trap_passed`'s own
rank-comparison logic, which `tests/test_eval.py` already covers with
synthetic in-memory records (§2) — it adds no new coverage once T2 is no
longer asserted to pass. In its place:
- `test_t2_baseline_is_expected_to_fail()`, decorated
  `@pytest.mark.xfail(strict=True, reason="personal-rag has no "
  "freshness/recency signal between two processed notes — see "
  ".specs/stale-trap-fixtures.md §4.5")`: runs T2's query (RunPod serverless
  vocabulary, per §4.5) through the real fixture store with the **default**
  filter active, and asserts `stale_trap_passed(records, expected,
  must_not)` is **True**. Because of `strict=True`, this assertion being
  **False** today is reported as `xfail` (expected, healthy); if a future
  change makes it **True**, pytest reports an **unexpected pass**, which
  `strict=True` turns into a **hard test failure** — forcing a human to
  look at §4.5 and decide whether the new passing behavior is a real
  improvement (e.g. a freshness signal was added) or another wording
  accident. This directly implements "strict xfail means T2 starting to
  pass is a test failure that forces a decision."
- No precondition test is needed for T2 — unlike T1/T3, T2's whole point is
  that *nothing* defends it, so there is no filter-disabled-vs-enabled
  contrast to prove is doing work. The single xfail assertion **is** the
  teeth: it fails loudly (as an unexpected pass) the moment the system
  starts doing something it doesn't do today.

### 6.4 Why real-model tests here, and the fallback

- **[Observed]** `AGENTS.md`'s own description of `make test-unit`: "fully
  offline: ... the store/backup/e2e tests run a real `ChromaStore` over a temp
  dir with a **fake** embedding model." Every existing test that builds a real
  `ChromaStore` (`test_end_to_end.py`, confirmed) uses `_FakeModel`, never the
  real MiniLM — so this plan's §6.2/§6.3 tests loading the real
  `sentence-transformers/all-MiniLM-L6-v2` would be the **first** `make
  test-unit` test to do a real model load.
- **[Observed]** `tests/conftest.py` excludes `tests/test_queries.py` from
  collection specifically because *that* file "needs a populated index and a
  real model download" (`AGENTS.md`). A cold CI runner with no HF cache would
  need to download MiniLM (~90 MB) the first time these new tests run — not
  "no network" in the strict sense `make test-unit` otherwise guarantees.
- **[Proposal]** Resolve by putting `test_eval_fixtures.py`'s real-model tests
  (§6.2, §6.3) behind the **same skip condition** `test_queries.py` already
  uses, if `tests/conftest.py` exposes a reusable marker/fixture for "needs a
  real model" — **[Not verified]**, the coding agent must open
  `tests/conftest.py` to check whether such a marker already exists or must be
  added. If no such mechanism exists yet, add a `pytest.ini`/`conftest.py`
  marker `requires_model` (skipped by default in CI's `make test-unit`
  unless the model is cached, consistent with how `test_queries.py` is
  excluded entirely from collection) — but **do not** exclude this file from
  collection the way `test_queries.py` is fully excluded, because §6.1's
  structural tests should still run in CI with zero cost; only the model-
  loading tests need gating.
- **[Alternative, rejected]** Using a fake dim-N model (like
  `test_end_to_end.py`'s `_FakeModel`) for the stale-trap teeth tests was
  considered and rejected: a fake model's embedding is a length-based
  bag-of-features with no semantic content, so it cannot be trusted to
  reliably favor the stale page's vocabulary over the current page's for
  T1/T3's precondition step (§6.3) — the entire thing under test (does the
  real model's ranking genuinely favor the stale wording before the filter
  removes it) would be unfalsifiable with a fake model. The real model is
  required because the precondition and the T2 baseline both depend on real
  semantic similarity, not a length-based proxy.

## 7. Not read / not verified

- **The three vault notes named in the brief** — **resolved for T1**: Vít
  verified "Logmanager Parser Generator Sync Refactor" against the vault on
  2026-10-08 and corrected T1's current-page fact accordingly (§4.1, §4.2).
  **T2/T3 remain as stated in the brief** ("Logmanager Alert and Parser
  Generator EKS Deployment on AWS", "Bedrock Structured Outputs vs the Alert
  Generator EBNF Grammar") and are confirmed by Vít to match the vault as
  written — no outstanding correction. This plan itself still has not opened
  any of the three notes directly; the T1 correction and the T2/T3
  confirmation are **[Observed, by Vít]**, not independently re-verified by
  this planning pass.
- **`src/rag/store/chroma_store.py`'s `ChromaStore` class body** — not opened
  this pass (only its import surface via `store/__init__.py` and its external
  call shape via `test_end_to_end.py`/`test_store.py` were read). §3.2(b)'s
  claim "nothing global/shared beyond `(index_path, name)`" is **[Inferred]**
  from those two test files successfully running independent instances, not
  from reading the constructor. Must be confirmed before implementation —
  if there *is* a module-level client cache keyed more coarsely, two
  `ChromaStore`s in the same process (the real golden-set store + the fixture
  store) could collide.
- **`config.yaml`'s actual `default_excluded_status` value** — **resolved**:
  `config.yaml:81` sets `default_excluded_status: ["needs-review"]`
  (`grep -n default_excluded_status config.yaml`, confirmed live during this
  amendment). §6.3's precondition tests rely on this value being present;
  no further check needed before implementation.
- **`tests/conftest.py`** — not opened; §6.4's "same skip condition" and
  "reusable marker" are **[Proposal]**, contingent on what's actually there.
- **`src/rag/reconciliation.py`'s `ReconciliationCatalog` constructor** — not
  opened; §5.1 assumes a "fresh, empty" catalog can be constructed ad hoc for
  a one-shot store build (as opposed to only via `ReconciliationCatalog.
  from_store(store, index_path)`, which is the one call site `.specs/
  keeping-it-true.md` §1(b) documents). **[Observed, keeping-it-true.md
  §1(b)]** only confirms `from_store()`; whether a bare/empty constructor path
  exists or whether the harness should call `from_store(fixture_store,
  tmp_dir / "chroma")` instead (simpler, and consistent with every other
  caller) needs a one-line check against `reconciliation.py` before writing
  `build_fixture_store`. **[Proposal, revised]**: default to
  `ReconciliationCatalog.from_store(fixture_store, tmp_index_path)` for
  consistency with the only documented call pattern, unless that module shows
  a more direct "always new" constructor is intended for this exact
  one-shot-build use case.
- **Whether `rag-drift`'s own test suite (`tests/test_drift.py`,
  per `.specs/keeping-it-true.md` §1 "Tests") asserts anything about
  `./chroma_db` directly that a parallel temp-directory fixture store could
  conceivably interfere with** — not opened. **[Inferred, low risk]**: since
  the fixture store never shares a path with `index_path`/`vault_path`, this
  should be a non-issue, but it is not independently verified.
- **Whether CI (`.github/workflows/ci.yml`) has HF model caching across
  runs** — not opened. Relevant to how expensive adding a real-model test is
  in CI wall-clock terms; the coding agent should check before deciding
  whether `--no-stale-trap-fixtures`-equivalent gating in CI is also warranted
  for `make eval` runs in CI (if `make eval` runs in CI at all — **[Not
  verified]** whether it does; `AGENTS.md`'s CI description lists `make lint`,
  `make typecheck`, `make test-coverage`, `make package-check` only, not
  `make eval`, so this is likely moot, but not confirmed).

## 8. Acceptance shape — mapped to this plan

- "`make eval` reports T1 and T3 passing and T2 as an expected-fail baseline"
  → §5.2's default-True `include_stale_trap_fixtures` wired through
  `rag.eval.main()`'s existing call to `evaluate()`; verify by running
  `.venv/bin/rag-eval` (or `make eval`) once post-implementation and reading
  the `STALE-TRAP 2/2 passed` line plus the separate
  `BASELINE (expected fail, not counted)` block showing T2 as FAIL, in the
  console output.
- "A test with teeth ... needs-review filter disabled ... fails" → §6.3's
  `test_t1_needs_review_trap_has_teeth()` / `test_t3_needs_review_trap_has_
  teeth()`, each proving an explicit precondition (stale page ranks #1 with
  the filter off) before asserting pass-with-filter / fail-without-filter —
  deterministic by construction, not dependent on embedding luck.
- "strict xfail means T2 starting to pass is a test failure that forces a
  decision" → §6.3's `test_t2_baseline_is_expected_to_fail()`,
  `@pytest.mark.xfail(strict=True)`; an unexpected pass is reported by pytest
  as a hard failure, forcing review against §4.5.
- "No change to the production index; `rag-drift` stays clean after `make
  eval`" → §3.3's argument (fixture harness never touches `vault_path` or
  `index_path`); verify by running `rag-drift` (against the real config)
  immediately before and after a `make eval` run and diffing the two reports —
  **[Proposal]** add this exact before/after `rag-drift` diff as a manual
  verification step in the coding agent's PR description, not as an automated
  test (an automated test would need a populated real index, which `make
  test-unit` doesn't have — this is a `make eval`-time human/CI-manual check,
  consistent with how `rag-drift` itself is verified per `.specs/
  keeping-it-true.md` §1's test list, which is pytest-only against fixtures,
  not the live Jetson index).
- "Keep the 45 existing golden rows and the main recall/MRR numbers
  unaffected" → §5.4 (no config change) + §5.2 (golden-JSONL scoring path
  untouched) + the existing `test_golden_set_wellformed` bound check
  (30-60 rows: 45 real + 0 new JSONL stale-trap rows = still 45, since this
  plan's 3 rows live in `eval_fixtures.py`, **not** appended to
  `golden_queries.jsonl` — a deliberate choice: the fixture rows' `expected`/
  `must_not` substrings only resolve against the fixture store's synthetic
  titles, so putting them in the real golden JSONL and running them against
  the real production store via `evaluate()`'s normal per-row loop would be
  wrong — they would never match anything in the real index. They **must**
  run only through `run_stale_trap`'s dedicated fixture-store path, which is
  exactly why `eval_fixtures.py` carries its own `STALE_TRAP_GOLDEN` list
  instead of three new lines in `golden_queries.jsonl`).

## 9. Files touched (summary)

| File | Change |
|---|---|
| `src/rag/eval_fixtures.py` | **New.** `FIXTURE_PAGES`, `STALE_TRAP_GOLDEN`, `build_fixture_store()`, `run_stale_trap()`. |
| `src/rag/eval.py` | `evaluate()` gains `include_stale_trap_fixtures=True` kwarg; merges `run_stale_trap(...)` rows into the `stale_trap` block. `main()` gains `--no-stale-trap-fixtures`. One doc-comment line. |
| `tests/test_eval_fixtures.py` | **New.** Structural tests (§6.1, fast/no model), harness pass test (§6.2, real model, gated per §6.4), two teeth tests (§6.3, real model, gated per §6.4). |
| `tests/conftest.py` | Possibly: add/confirm a `requires_model`-style marker (§6.4) — contingent on what's already there; **not read yet**. |
| `tests/eval/golden_queries.jsonl` | **Unchanged.** Stays at 45 rows (§8). |
| `config.yaml` / `config.personal.yaml` | **Unchanged** (§5.4). |

## 10. Known risks

- **Real-model test cost in CI** — §6.4's open question; mitigated by gating,
  not solved outright until `tests/conftest.py` is actually read.
- **T1/T3 precondition might not hold on the first wording attempt** — §6.3's
  precondition step (stale page must rank #1 with the filter disabled) could
  fail with the first-draft fixture text if MiniLM doesn't separate the
  stale/current vocabulary as cleanly as expected. Per Vít's instruction this
  is resolved by **sharpening the fixture wording** (more topic-specific,
  more lexically distinct from the current page), never by weakening the
  assertion — expect at least one iteration on T1/T3's exact page text during
  implementation.
- **`ChromaStore` global-state assumption (§3.2b)** — unverified against the
  actual class body; if wrong, could cause flaky cross-contamination between
  the fixture store and the real golden-set store within one `evaluate()`
  call. Must be checked first, cheaply (reading one file).
- **T2 is a single baseline row** — if its query/wording is later changed for
  any reason, the xfail's `reason=` text and §4.5's rationale should be
  re-checked together so the two don't drift apart; a baseline whose intent
  isn't legible from the test file alone is easy to accidentally "fix" by
  weakening it instead of treating an unexpected pass as a signal.
</content>
