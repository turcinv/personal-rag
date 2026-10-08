# Keeping It True — personal-rag indexing-side plan

Plan only. No code written, no index touched, no reindex run. Scope: drift
check, `needs-review` handling in retrieval, and stale-trap eval questions.
Mirrors the vault's "Keeping It True" design (planning#3852); the vault side
(warn-only validator hook, Verified-age audit) is already done.

Each claim below is tagged **[Observed]** (file+line/commit/command output),
**[Inferred]** (reasoned from observed facts), or **[Proposal]** (what to build).
A "not read / not verified" list is at the end.

---

## 0. Verified facts the plan rests on

### Index & retrieval seams
- **[Observed]** Chunk IDs are content-hashed: `stable_id(*parts)` returns
  `sha256("::".join(parts))` and is called as
  `stable_id(rel_path, section_index, chunk_index, chunk)` —
  `src/rag/chunking.py:154` and `src/rag/extractors/markdown.py:96`. Identical
  chunk text ⇒ identical ID ⇒ idempotent incremental indexing.
- **[Observed]** `query.search()` is the single retrieval seam. CLI
  (`rag.query.main`), eval (`rag.eval.evaluate` → `search`,
  `src/rag/eval.py:106`), API and MCP all call it. `src/rag/query.py:118`.
- **[Observed]** Metadata filtering is backend-neutral: callers build a
  `RetrievalFilter` (`src/rag/retrieval.py:40`), `ChromaStore.compile_where`
  turns it into a Chroma `$eq`/`$and` where-dict in exactly one place
  (`src/rag/store/chroma_store.py:33`). `status` is already a first-class
  scalar field (`SCALAR_FIELDS`, `src/rag/retrieval.py:30`) and the CLI already
  has `--status` (`src/rag/query.py:355`).
- **[Observed]** Provenance is enforced: `ensure_compatible()` aborts on a
  fingerprint mismatch and `open_store()` calls it on every query path
  (`src/rag/provenance.py:155`, `src/rag/query.py:174`). The fingerprint is
  built from embedding model/revision/dimension, chunker version, chunk sizes,
  metric, corpus profile — **not** from the metadata schema
  (`IndexProvenance`, `src/rag/provenance.py:25`).

### Status in chunk metadata — the key unknown, now resolved
- **[Observed]** `status` **already reaches chunk metadata**. The live Markdown
  extractor writes it: `status = str(meta.get("status") or "")` and
  `"status": status` into every chunk's metadata dict
  (`src/rag/extractors/markdown.py:67,90`).
- **[Observed]** It is present in the live on-disk index now. Direct sqlite peek
  of `chroma_db/chroma.sqlite3` `embedding_metadata` where `key='status'`:
  `processed` 38 411, `needs-review` 2 179, `active` 671, `archived` 75,
  `inbox` 18, and `""` (empty) 214 798. (The empty bucket is the book/resource
  JSON corpus, which carries no vault status.)
- **[Inferred]** Therefore adding `needs-review` handling to retrieval is a
  **query-time / metadata-only** change: no re-embed, no provenance bump, no
  metadata backfill. The `needs-review` chunks are already selectable today via
  `rag-query --status needs-review`. This is the smallest-blast-radius branch
  and it is available for free.
- **[Observed]** `build_vault_index.py` is a *separate* JSONL path
  (`classification_status` field, `src/extractor/build_vault_index.py:168`) that
  feeds the FTS SQLite DB, **not** the Chroma vector index. The Chroma index is
  built only by `rag.indexer` → `iter_sources` → `extract_md_file`. The drift
  check and `needs-review` work target the Chroma path.

### Anti-wipe / reconciliation
- **[Observed]** Source-scoped reconciliation lives in
  `ReconciliationCatalog` (`src/rag/reconciliation.py`): it pages existing
  metadata into a temp SQLite catalog, and `classify_and_mark(ids, metas)`
  returns `(new_indices, updated_indices)` by comparing the content-hash
  `chunk_id` and the serialized metadata JSON (`src/rag/reconciliation.py:150`).
- **[Observed]** The 0-files anti-wipe guard: `rag.indexer.main` raises if every
  source reports 0 files while the index is non-empty
  (`src/rag/indexer.py:243`), plus per-source prune thresholds
  (`prune_max_fraction` 0.25 / `prune_max_chunks` 10 000) that block large
  deletions unless `--allow-large-prune` (`src/rag/index_manifest.py:112`).

### Sync to the Jetson (the `<<SYNC>>` the prompt asked for)
Verified live, 2026-10-08:
- **[Observed]** The macOS **code** checkout `~/Documents/personal-rag` is a git
  repo, remote `origin = github.com/turcinv/personal-rag`, branch `main`, HEAD
  `6a2f2ed`, with uncommitted local work (classifier: `src/rag/classifier.py`,
  `tests/test_classifier.py`, edits to `markdown.py`/`chunking.py`/`json_doc.py`/
  `pdf.py`/`Makefile`/`.gitignore`). (`git` in the repo.)
- **[Observed]** The macOS **vault** `~/Documents/personal_knowledge/Career
  Knowledge Base` is a git repo on `main`, HEAD `70f5259e`, remotes
  `fedora = fedora:git/career-knowledge-base.git` and
  `jetson = jetson:git/career-knowledge-base.git`. Mac HEAD is **6 ahead / 0
  behind** cached `fedora/main` (`97e53957`) — the unpushed KB-audit commits.
- **[Observed]** The **Jetson** (`gpu-01`, reachable over Tailscale/SSH):
  - vault at `~/personal_knowledge/Career Knowledge Base` is a git repo on
    `main`, HEAD `9428b5e`, **working tree clean**, and has **no git remote**
    configured (`git remote -v` empty). Its reflog shows HEAD was last set by
    `pull --ff-only /home/turcinv/git/career-knowledge-base.git main` — i.e. it
    pulls from a **local bare repo on the Jetson**, not from fedora.
  - `~/personal-rag` code is a git repo, HEAD `7e66343`.
- **[Observed]** Jetson vault HEAD `9428b5e` is an ancestor of mac HEAD;
  mac `main` is **494 commits ahead** of it and `fedora/main` is **488 ahead**.
  Jetson vault's last commit date is **2026-08-18** — the indexed vault on the
  Jetson is roughly **7 weeks stale**.
- **[Observed]** No local macOS automation syncs any of this: `crontab -l`
  empty, no matching `~/Library/LaunchAgents`. The documented procedure is the
  manual `docs/SOP-jetson-sync.md` (git-pull OR rsync for code+vault; the
  `make sync-to-jetson` generation transfer is for extraction artifacts only,
  never the vault or Chroma store — `src/extractor/sync.py:1`).
- **[Inferred]** This is live, real drift of exactly the kind the plan must
  catch: the Jetson is serving an index built from a vault ~494 commits behind
  the source of truth, and nothing automatically closes the gap. The Jetson
  vault having **no remote** means the SOP's "git pull" path cannot run there as
  written — someone must push into the Jetson's local bare repo, or the Jetson
  must gain a `fedora` remote. **This is a finding to surface to Vít, not
  something this plan changes.**

---

## 0b. Vault sync to the Jetson  [Proposal, grounded in Observed facts]

The drift check (§1) *detects* staleness; this section *prevents* it by closing
the vault→Jetson→index loop on a schedule, so the served index tracks the vault
without a human remembering the SOP. Plan only — nothing here is installed.

### Facts this rests on
- **[Observed, spec §0]** The Jetson vault working copy has **no git remote**;
  its reflog shows the last update was `pull --ff-only
  /home/turcinv/git/career-knowledge-base.git main` — i.e. it fast-forwards from
  a **local bare repo on the Jetson** at `~/git/career-knowledge-base.git`. Vault
  HEAD `9428b5e` (2026-08-18), **494 commits behind** macOS `main`. Jetson code
  checkout HEAD `7e66343`.
- **[Observed by Vít, 2026-10-08]** On the Jetson: no user crontab, no user
  systemd timers, no system timer or `/etc/cron.d` entry for sync/index;
  `docker ps` showed no running containers; login shell is fish.
- **[Observed, macOS, 2026-10-08]** The vault repo pushes to remote
  `jetson = jetson:git/career-knowledge-base.git`; remote-tracking `jetson/main`
  = `7153c329` dated **2026-10-08 10:28** and mac HEAD is **0 ahead / 0 behind**
  it — so the **bare repo is currently up to date** to the last push. Primary hub
  is `fedora`; pushing to `jetson` is a manual habit, **not enforced**:
  (`git -C <vault> config core.hooksPath` → `.githooks`, which contains **only a
  `pre-commit` hook** — no `pre-push`/`post-commit` mirror to jetson or fedora).
- **[Observed]** The index writer lock is an advisory `flock` on
  `<index_path>.writer.lock` *beside* the Chroma dir, self-healing on process
  death (`src/rag/locking.py:1,10,84`); `rag.indexer` acquires it for any
  non-dry-run run (`src/rag/indexer.py:286`).
- **[Not verified]** The bare repo's own HEAD on the Jetson host; whether the
  Jetson can reach `fedora` over ssh (see §6).

### Recommended mechanism — systemd **user** timer on the Jetson  [Proposal]
A single systemd *user* timer + service unit on the Jetson, owned by `turcinv`
(no root), that does, in order:
1. `git -C ~/personal_knowledge/Career\ Knowledge\ Base pull --ff-only
   ~/git/career-knowledge-base.git main` — **[Observed]** this exactly matches
   the mechanism the reflog shows is already in use, so it is the smallest change
   from today's reality (nothing new learns a new path);
2. **only on pull success AND only if HEAD changed**, run `make jetson-index` in
   `~/personal-rag`.
Capture `git rev-parse HEAD` before/after; skip the index step when unchanged so
a quiet period costs nothing. Everything runs as the existing user — **[Proposal]**
no `.env` is read, copied, or referenced by the units.

**Why this over the alternatives:**
- **vs. pulling from `fedora` directly** — attractive because it removes the
  "did I remember to `git push jetson`?" dependency (**[Observed]** that push is
  unenforced). But it **[Not verified]** requires the Jetson to reach `fedora`
  over ssh/Tailscale and a `fedora` remote to be added on the Jetson vault — two
  unknowns and one state change. Recommend keeping the pull **from the local bare
  repo** (matches the reflog, zero new reachability assumptions) and fixing
  upstream freshness separately (next subsection), rather than coupling the two.
- **vs. a launchd rsync from macOS** — pushes bytes instead of commits, bypasses
  git history, risks copying a dirty tree, and makes macOS responsible for the
  Jetson's state (the exact fragility the SOP's §4 `/tmp` wipe warning guards
  against, `docs/SOP-jetson-sync.md`). More moving parts, less safe. Rejected.
- The user timer has the **fewest moving parts** that still gives scheduling,
  journald logging, and `systemctl --user status` visibility, and needs no root.

### Failure handling  [Proposal]
- `--ff-only` means a **non-fast-forward** (diverged) pull **fails** rather than
  merging; the service treats a non-zero `git pull` as fatal: **log to journal,
  exit non-zero, never `reset`/`stash`/`--force`/`checkout -f`.**
- A **dirty** working copy: pre-flight `git status --porcelain`; if non-empty,
  **stop before pulling**, log, exit non-zero (a dirty Jetson vault is unexpected
  — **[Observed]** it was clean on 2026-10-08 — and silently stashing it would
  lose state).
- The **index step never runs after a failed/short-circuited pull** (shell `&&`
  chaining / explicit exit-code gate in the service `ExecStart` script).
- **Concurrency with a manual `make jetson-index`:** **[Observed]** the indexer
  already serializes writers via the advisory `flock` on
  `<index_path>.writer.lock` and fails fast with `IndexLockedError` if held
  (`src/rag/indexer.py:291`), so two indexers cannot corrupt the store. **[Proposal]**
  pass `--wait-for-lock <seconds>` (the indexer already supports it,
  `src/rag/indexer.py:146`) in the timer's index step so a run colliding with a
  manual index **waits** rather than failing the timer; a modest timeout (e.g.
  600 s) keeps the timer from hanging indefinitely.

### Upstream freshness — detecting a stale bare repo  [Proposal]
The timer keeps the Jetson vault = bare repo; it does **not** help if **macOS
forgot to push to `jetson`** (then the bare repo itself lags). **[Observed]** the
push is unenforced today. Recommendation, smallest moving part first:
- **Recommended:** have `make drift` §1(a) compare the **Jetson vault** against
  **`fedora/main`** (the real hub), not just against its own bare upstream — so a
  stale bare repo shows up as "behind fedora/main" the moment drift runs after a
  sync. This needs the Jetson vault to have a read-only view of `fedora` **[Not
  verified: reachability]**; if it cannot reach `fedora`, drift falls back to
  reporting "upstream not configured" (already in §1a) and the staleness is
  surfaced instead by the macOS side below.
- **Also recommended (macOS side, enforces the push):** add a `pre-push` or
  `post-commit` mirror to the vault's existing `.githooks/` flow
  (**[Observed]** `core.hooksPath=.githooks`, only `pre-commit` present today)
  that pushes `main` to `jetson` (and `fedora`) so "I forgot to push to jetson"
  cannot happen. This is a **vault-repo** change, out of scope for *this*
  (personal-rag) spec — **flag to Vít as a one-line vault follow-up**, do not
  implement here.
- **Not recommended:** making the Jetson timer pull from `fedora` *instead of*
  the bare repo — see the mechanism trade-off above (unverified reachability +
  a remote-add state change).

### Code checkout + image mismatch  [Proposal]
- **[Proposal]** The timer updates the **vault + index only**, not the code.
  A `git pull` of `~/personal-rag` plus `make build-jetson` (image rebuild) stays
  a **separate manual step** — **[Observed]** the image bakes the source
  (`docs/SOP-jetson-sync.md` step 5), so an automated code pull without a rebuild
  would run new source against a stale image, and auto-rebuilding an image from a
  timer is exactly the kind of heavy, failure-prone action that should stay
  human-initiated. Do **not** automate image builds.
- **[Proposal]** Notice a code/image mismatch cheaply: have the sync service log,
  each run, the Jetson `~/personal-rag` `git rev-parse --short HEAD` vs the
  running image's baked commit (if the image records one; **[Not verified]**
  whether it does — see §6). If they differ, journal a WARNING. No action taken —
  just visibility, consistent with the drift philosophy.

### Wire the drift check into the timer  [Proposal]
- **[Proposal]** After the index step, run `rag-drift` (§1) and send its exit
  code to the journal, so a sync that "succeeds" but still leaves drift (e.g. the
  bare repo was stale, or a note failed to extract) is **visible** in
  `journalctl --user -u <service>` without a human running anything. The drift
  run is read-only (§1) and embedding-free, safe to chain after the index.

### One-time catch-up (first run)  [Proposal / partly Observed]
- **[Inferred]** The first scheduled run fast-forwards ~494 commits of vault
  history and then indexes the net new/changed notes (**[Observed]** the SOP
  cites "~250+ new notes" order of magnitude is **not** stated there verbatim;
  the SOP's expected magnitudes are "~3,000 vault markdown files, ~260 JSON
  docs" at *full* enumeration — `docs/SOP-jetson-sync.md` step 6 — **not** a
  delta count, so the catch-up delta is **[Not verified]**).
- **[Observed]** The run is incremental: content-hash chunk IDs mean unchanged
  chunks are skipped and only changed/new ones embed (`src/rag/chunking.py:154`),
  so this is **not a full re-embed**.
- **[Observed]** The 0-files anti-wipe guard **applies** to this run exactly as
  to any `jetson-index`: `rag.indexer.main` raises if every source reports 0
  files while the index is non-empty (`src/rag/indexer.py:243`), and source-scoped
  prune thresholds still gate large deletions
  (`src/rag/index_manifest.py:112`). The timer adds no `--allow-large-prune` /
  `--allow-empty-source-prune` override.
- **[Observed]** Expected incremental sync+reindex time per the SOP is "~5 min"
  for a routine delta; a full re-embed (not this case) is "considerably longer"
  (`docs/SOP-jetson-sync.md` "Time"). The 494-commit catch-up is larger than a
  routine delta but still metadata/incremental, **[Not verified]** exact duration.

### Where the units live + install/uninstall  [Proposal]
- **[Proposal]** Unit files live **in this repo** under `deploy/jetson/`:
  `rag-sync.service` (oneshot `ExecStart` = a small `deploy/jetson/rag-sync.sh`
  that does pull → conditional index → drift) and `rag-sync.timer` (e.g.
  `OnCalendar=hourly` or a few times/day, `Persistent=true` so a missed run
  catches up). The script is POSIX `sh`/bash (invoked explicitly, not relying on
  the fish login shell).
- **[Proposal]** Two make targets: `make jetson-sync-install` (copies the two
  units to `~/.config/systemd/user/`, `systemctl --user daemon-reload`,
  `enable --now rag-sync.timer`) and `make jetson-sync-uninstall`
  (`disable --now`, remove units, reload). Both run **on the Jetson**.
- **[Proposal]** The units **must not** reference, read, or copy the Jetson
  `.env`; the indexer reads its own env/config on the host as it does for a
  manual `make jetson-index` (constraint honoured).

---

## 1. Drift check — `rag-drift` / `make drift`  [Proposal]

A **report-only** command. New module `src/rag/drift.py` (one-line docstring,
timezone-aware datetimes), entry point `rag-drift = "rag.drift:main"` in
`pyproject.toml` `[project.scripts]`, and a `make drift` target. Embeds nothing;
opens the store read-only; **never** mutates the index.

### 1(a) Source checkout vs the vault's upstream
- **[Observed]** The indexed source is `config["vault_path"]`
  (`src/rag/indexer.py:171`), and that path is itself a git repo with a `fedora`
  remote.
- **[Proposal]** When `vault_path` is inside a git work tree:
  - run `git -C <vault> fetch --quiet fedora` (configurable remote, default
    `fedora`), then report `git rev-list --left-right --count HEAD...fedora/main`
    (ahead/behind) and the dirty-file count from `git status --porcelain`.
  - Drift if behind > 0, or dirty > 0 (uncommitted source the index can't match
    a committed state to), or ahead > 0 **with** a warning (local commits not on
    upstream — the host leads, which on the Jetson would be unexpected).
  - When `vault_path` is **not** a git work tree (e.g. a Jetson rsync target):
    say so explicitly and skip this sub-check (exit contribution = "unknown, not
    a git repo", not a false "no drift"). **[Observed]** this case is real — the
    Jetson vault has no remote, so even in-git it can't compare to `fedora/main`;
    the command must detect "git repo but no `<remote>/main`" and report
    "upstream not configured" rather than erroring.
- **[Proposal]** Use `subprocess.run([...], check=False)` with argument lists
  (no shell); treat any git failure as "unknown", never as "clean".

### 1(b) Source files vs the index (the core check)
Reuse the existing reconciliation classifier **read-only** — do not re-embed.
- **[Observed]** `iter_sources(config, vault_path, max_chars, overlap)` yields
  the exact same `Source` objects the indexer uses, each with `.files` and an
  `.extract` callable (`src/rag/extractors/__init__.py:171`). `extract_md_file`
  returns `(ids, docs, metas, error)` where `ids` are the content-hash chunk IDs
  — computed **without embedding** (`src/rag/chunking.py:154`).
- **[Observed]** `ReconciliationCatalog.from_store(store, index_path)` pages the
  current index metadata into a temp catalog; `classify_and_mark(ids, metas)`
  tells new-vs-changed per file, and `source_counts()` / `iter_stale_batches()`
  expose stale (indexed-but-no-longer-in-source) chunks
  (`src/rag/reconciliation.py:120,150,232,266`).
- **[Proposal]** `rag-drift` for the Markdown source (optionally all sources):
  1. open the store read-only, build a `ReconciliationCatalog` from it;
  2. for each file, call `source.extract(path)` (chunk + hash only, **no model
     passed**, so no embedding) and `classify_and_mark` to mark seen and collect
     counts;
  3. report three buckets:
     - **missing from index**: files whose chunks are all `new_indices`
       (present in source, absent in index);
     - **stale in index**: `source_counts().stale` + `iter_stale_batches` —
       indexed chunks whose file/content is no longer produced by the source;
     - **content changed**: files with `updated_indices` (same chunk_id absent
       but file seen with different content ⇒ a changed chunk hashes to a new
       ID, so this surfaces as new+stale for that file; report at file
       granularity by path).
  - **[Inferred]** Because IDs are content hashes, "edited one note" shows up as
    that note's old chunk(s) stale + new chunk(s) missing — exactly the acceptance
    shape ("reports exactly that note").
  - **[Proposal]** The catalog uses the model **only** inside `index_file_chunks`
    for actual upserts; `rag-drift` never calls that path, so it is embedding-free
    and safe to run against the live Jetson index without a GPU spin-up.
- **[Proposal]** `--source <id>` to scope (default: markdown:vault), `--json`
  for machine output, human table otherwise.

### Exit code & cron-readiness  [Proposal]
- Exit `0` = no drift and every requested sub-check actually ran. Exit `1` =
  drift: any drift bucket non-empty or the git sub-check reports behind/dirty.
  Exit `2` = **could not verify**: no drift was observed, but at least one
  sub-check could not run — the git check is `unknown` (not a git work tree, or
  upstream unreachable) or a requested source was not `available` for
  enumeration (e.g. the Jetson vault-mount failure). A `2` prints
  `COULD NOT VERIFY — <reasons>` so a cron does not read an unverifiable run as
  green. Drift wins: if drift is observed, the exit is `1` even when another
  sub-check is `unknown`. This lets a later cron (`cron_add script=…` or a
  Jetson systemd timer) alert on both drift and could-not-verify. The command
  itself schedules nothing — scheduling is out of scope here.
- **[Observed]** The git sub-check runs `git fetch --quiet <remote>`, which
  updates the local remote-tracking refs (`refs/remotes/<remote>/*`) — so the
  command is **not strictly read-only against the local git metadata**, though
  it never mutates the index, the store, or the working tree. The fetch is
  otherwise harmless and is what makes the ahead/behind comparison meaningful.
- **[Proposal]** Read-only guarantee: `rag-drift` takes **no** `IndexWriterLock`
  (consistent with `--dry-run` in `indexer.py`, which is lock-free —
  `src/rag/indexer.py:286`), opens the store for reads, and the temp catalog is
  deleted on context exit (`ReconciliationCatalog.close`,
  `src/rag/reconciliation.py:110`).

### Tests  [Proposal]
`tests/test_drift.py`: (i) in-sync fixture (git forced OK) → empty buckets,
exit 0; (ii) add a file not in the index → reported missing, exit 1; (iii)
change one note's body → that note reported, exit 1; (iv) non-git `vault_path`
→ git sub-check "unknown", no crash, exit 2 ("COULD NOT VERIFY"); (v) an
unavailable/unknown source → exit 2; (vi) drift present with git unknown → exit
1 (drift wins). Reuse the store doubles already in `tests/test_indexing.py` /
`tests/test_provenance.py`.

---

## 2. `needs-review` handling in retrieval  [Proposal + decision needed]

### Current state  [Observed]
- `status` is on every vault chunk and `needs-review` is a live value (2 179
  chunks). `search()` already supports `--status needs-review` /
  `RetrievalFilter(status=...)`. There is **no** default exclusion today: a
  `needs-review` note is returned in normal search exactly like a `processed`
  one. The CLI/API/MCP never set a default status filter.

### Team-KB design to mirror (2026-10-08)
Pages past review are excluded from default search, returned only with an
explicit `include_expired`-style flag, and carry a structured flag when returned.

### For this personal KB — decided 2026-10-08 (Vít)
Option (b) is chosen: `needs-review` is excluded from default search via a
post-filter in `query.search()`; an opt-in flag — `--include-unreviewed` (CLI) /
`include_unreviewed` (API, MCP) — disables the exclusion for that call; returned
records carry a structured `unreviewed` marker; no re-embed, no provenance
change. Revisit if the stale-trap eval (§3) shows the marker alone is enough.

Two framings differ: the team design is about *review expiry* ("past review"),
whereas this KB's `status: needs-review` means *not yet processed/cleaned*. The
behaviour the user wants is the same shape (don't serve unreviewed notes as
authoritative by default; allow opt-in; mark them when returned), so the
mechanism below fits either reading.

**Smallest-blast-radius option (recommended) — query-time default filter.**
Because `status` is already indexed and filterable, no index change at all:
- **[Proposal]** Add one config key, e.g. `default_excluded_status:
  ["needs-review"]` (empty/omitted ⇒ today's behaviour, byte-identical).
- **[Proposal]** In `search()` (the single seam), when the caller supplies no
  explicit `status` constraint and no `include_unreviewed`-style override, apply
  a **negative** status filter. **[Observed] blast-radius caveat:** Chroma 0.6.3
  `where` supports `$ne`/`$nin`, but the current `compile_where` only emits
  `$eq`/`$and` (`src/rag/store/chroma_store.py:40`). So this needs either
  (a) extend the neutral filter + compiler with an "exclude these status values"
  clause (one new clause type, still one compile site), or
  (b) **[Observed]** mirror the existing `tags` pattern — a **post-filter** in
  `search()` over the returned records' `status` metadata
  (`src/rag/query.py:258` shows the tags post-filter it would copy), widening
  the dense pool like `tag_fetch_k` so the post-filter has candidates. Option
  (b) touches only `query.py`, adds no store-dialect surface, and composes with
  rerank/hybrid the same way tags do — **recommend (b)**.
- **[Proposal]** An explicit opt-in — `rag-query --include-unreviewed` /
  `include_unreviewed: true` in the API/MCP request — disables the default
  exclusion for that call. The flag works identically for CLI, API, MCP, eval
  because it is resolved inside `search()`.
- **[Proposal]** Structured marker when a `needs-review` chunk IS returned
  (under the opt-in): `status` is already in each record's metadata; add a
  derived boolean the formatters surface, e.g. `record["unreviewed"] =
  (status in default_excluded_status)`, and have the CLI print a `⚠ unreviewed`
  tag and the API include it in the JSON. No new storage.

**Trade-off stated.** Option (b) is a post-filter, so like tags it is
best-effort within the fetched pool: a query that would only match
`needs-review` chunks could under-return once they are excluded (acceptable —
that is the point). Widening the pool (`fetch_k`) mitigates it for mixed
queries. Option (a) ($ne/$nin) is exact at the store level but adds a new clause
type to the one compile site and must be covered in the store tests.

**Rejected: a metadata backfill or re-embed.** Not needed — status is already
present. **Rejected: a provenance bump.** The metadata schema is not part of the
provenance fingerprint (`src/rag/provenance.py:25`), and we add no new indexed
field, so no generation change and no `ensure_compatible` abort.

### Tests  [Proposal]
`tests/test_query_search.py` additions: (i) default search excludes a
`needs-review` chunk that a `processed` chunk of the same topic does not; (ii)
`include_unreviewed=True` returns it, carrying the structured marker; (iii)
explicit `--status needs-review` still works (opt-in path unaffected); (iv)
`default_excluded_status: []` is byte-identical to today.

---

## 3. Stale-trap questions in `make eval`  [Proposal]

### Current eval format  [Observed]
- Golden set is JSONL at `tests/eval/golden_queries.jsonl`, one object per line:
  `{"query": ..., "expected": [substrings], "kind": "vault"|"resource"}`
  (`src/rag/eval.py` loader at `:48`; 45 lines today).
- Scoring: a query "hits" if any `expected` substring appears (case-insensitive)
  in a returned chunk's `title` **or** `path` metadata (`is_hit`,
  `src/rag/eval.py:66`). Metrics are recall@5, recall@10, MRR, aggregated
  overall and **per `kind`** (`aggregate` + `by_kind`, `src/rag/eval.py:85,138`).

### Why the current format can't score a stale-trap
- **[Observed]** A stale-trap needs the *wrong* note to be penalised, but
  `is_hit` only rewards a positive substring match — there is no notion of a
  "must-not-return" note or of ranking a current note above a superseded one.
- **[Proposal]** Extend the record schema with an optional `kind: "stale-trap"`
  and a `must_not` list (substrings of the superseded / `needs-review` note's
  title or path). Score a stale-trap query as **passed** iff the first hit is an
  `expected` (current) note **and** no `must_not` note appears above it (ideally
  not in the top-k at all). Keep `expected`/`is_hit` semantics for existing
  rows; `must_not` defaults to empty so all 45 current rows are unaffected.
- **[Proposal]** Report stale-traps **separately**: because `by_kind` already
  partitions by `kind`, a `kind: "stale-trap"` row set is reported on its own
  line automatically; add an explicit `stale_trap` summary block
  (passed/total) so a regression is obvious. `make eval` output and the JSON
  (`--out`) both carry it.
- **[Proposal]** Wire it so a stale-trap failing does **not** silently drag the
  main recall numbers — keep it a distinct section, consistent with the
  acceptance shape ("reports them separately").

### Candidate note pairs — for Vít to confirm (DO NOT assume)  [Proposal]
A stale-trap needs a *current* note and a *superseded or `needs-review*` note
that answer the same question differently. I have **not** mined the vault for
real contradictions in this plan (that is content work, and the rule is to not
invent vault content). Shape of what to confirm, grounded in what the eval
already covers and what the index shows:
1. **A `needs-review` vs `processed` pair on one topic.** 2 179 `needs-review`
   chunks exist; pick one whose topic also has a `processed` note, where the
   `needs-review` note states something the processed note corrects. (Candidate
   source: run `rag-query --status needs-review "<topic>"` for a few golden-set
   topics and eyeball.)
2. **A `Verified <date>` time-sensitive note vs an older note** making the
   now-outdated claim (the vault's Verified-age audit identifies these; the
   superseded one is the trap).
3. **A superseded SOP/decision** where a newer note records the change and an
   older note still describes the old procedure.
- **[Proposal]** For each confirmed pair, add one JSONL row:
  `{"query": "<the question>", "expected": ["<current note title>"],
    "must_not": ["<superseded/needs-review note title>"], "kind": "stale-trap"}`.
- **[Proposal]** 2–3 rows as the prompt asks; list the exact note titles in the
  row so the trap is auditable.

### Tests  [Proposal]
`tests/test_eval.py` additions: a `kind: "stale-trap"` row with a synthetic
store double where the current note ranks above the `must_not` note → passed;
invert the ranking → failed; confirm existing rows' metrics are unchanged.

---

## 4. Constraints check (from the prompt)
- **[Observed→honoured]** Never touch/sync the Jetson `.env`: nothing here reads
  or writes it; drift's git fetch is read-only.
- **[Honoured]** No full re-embed is required by any item; §2 is explicitly
  metadata-free and §1/§3 never embed. Any future need to re-embed would be
  called out — none is.
- **[Honoured]** `query.search()` stays the single seam: the `needs-review`
  default and opt-in are resolved inside it, so CLI/API/MCP/eval share them.
- **[Proposal]** Every new module (`drift.py`) starts with a one-line docstring
  and uses `datetime.now(UTC)` (timezone-aware), per repo convention.

## 5. Suggested build order (for the later implementation phase)
1. `rag-drift` §1 (self-contained, read-only, highest value — it surfaces the
   Jetson's 494-commit staleness today). Entry point + `make drift` + tests.
2. §2 `needs-review` query-time default (option b, post-filter) + opt-in flag +
   structured marker + tests.
3. §3 eval schema extension + 2–3 confirmed stale-trap rows + tests.
4. §0b Jetson sync units — **after** §1, because the timer chains `rag-drift`
   and because drift (behind-`fedora/main`) is how upstream-freshness is
   surfaced. `deploy/jetson/` units + `rag-sync.sh` + `make jetson-sync-install`
   / `jetson-sync-uninstall`. Verify on the Jetson: a `git push jetson main` on
   macOS is reflected in the vault + index within one timer period and
   `make drift` exits 0; a dirty/diverged tree fails visibly and skips indexing.
   The vault-repo `.githooks` push-mirror (upstream freshness, macOS side) is a
   **separate one-line follow-up for Vít in the vault repo**, not part of this
   personal-rag spec.
Each lands with `make test-unit` green and `make lint` clean (ruff select is
E9/F-only today — `pyproject.toml:[tool.ruff.lint]`). §0b ships no Python under
`src/`, so its "tests" are the Jetson acceptance checks above, not pytest.

---

## 6. Not read / not verified
- **`src/rag/mcp/server.py` and `src/rag/api/` route handlers** — not opened
  this pass. I verified the MCP/API go through `search()` from the module
  docstrings and `query.py` (`[Observed]` the seam), but I have **not**
  confirmed how each surfaces a per-record flag, so §2's "structured marker in
  the API/MCP response" is **[Proposal]**, not Observed, for those two layers.
- **`src/rag/lexical.py`** (hybrid BM25) — not read; I have not verified whether
  a `needs-review` post-filter composes with a lexical-only hit that has no
  `status` in its record. To check before implementing §2 with `hybrid=True`.
- **`tests/test_eval.py` internals** — not opened; §3's test sketch assumes the
  same store-double pattern as the other suites but is unverified.
- **Chroma 0.6.3 `$ne`/`$nin` support** — stated from general Chroma knowledge,
  **not** verified against this pinned version. §2 recommends the post-filter
  (option b) precisely so the plan does not depend on this.
- **Jetson index contents vs its vault** — I verified the Jetson *vault* git
  state and staleness, but did **not** run `rag-stats`/`rag-drift` on the Jetson
  index itself (no code to run yet; running anything there is out of scope for a
  plan). The 494-commit staleness is Observed from git; the resulting index
  drift is **[Inferred]** from it.
- **`docker-compose.jetson.yml` mounts** — not re-read this pass; the SOP's
  `/tmp` wipe warning (`docs/SOP-jetson-sync.md` step 4) is quoted from the SOP,
  not re-verified against the compose file.

### §0b (Jetson sync) — additional not-verified items
- **The Jetson bare repo's own HEAD** (`~/git/career-knowledge-base.git`) — not
  inspected; I verified macOS `jetson/main` = `7153c329` (2026-10-08 10:28) and
  that mac HEAD is 0/0 against it, so the bare repo is current *to the last
  push*, but I did not read the bare repo's HEAD on the host directly.
- **Whether the Jetson can reach `fedora` over ssh/Tailscale** — not verified.
  This gates the recommended upstream-freshness check (drift comparing the
  Jetson vault to `fedora/main`) and the rejected "pull from fedora directly"
  alternative. Must be confirmed before implementing either.
- **Whether the Jetson vault has (or needs) a `fedora` remote** — Observed it has
  **no** remote at all; adding one is a state change deliberately left out of
  this plan.
- **Whether `personal-rag:jetson` records its baked source commit** — the §0b
  code/image-mismatch WARNING assumes the running image exposes a commit to
  compare against the checkout; not verified that it does (would need to read
  `Dockerfile.jetson` / image labels).
- **The 494-commit catch-up duration and exact new-note delta** — Inferred to be
  incremental (content-hash IDs, Observed) and larger than the SOP's "~5 min"
  routine delta, but the exact count/time is not measured. The addendum's
  "~250+ new notes" figure is **not** found verbatim in `docs/SOP-jetson-sync.md`
  (which cites full-enumeration magnitudes, not a delta), so it is treated as
  unverified.
- **No systemd user timers / crontab / containers on the Jetson** — Observed
  **by Vít**, not re-run by me this pass (the addendum forbids state changes and
  I relied on his read-only check rather than duplicating it).
