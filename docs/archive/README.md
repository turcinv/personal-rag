# docs/archive — historical snapshots, NOT current reference

Everything in this directory is a **point-in-time artifact**: a review, a fix
report, or an implementation plan that has since been executed or abandoned. It
is kept because it records *why* decisions were made, not *what is true now*.

**Do not follow instructions from these files.** Several describe a system that
no longer exists, and at least two contain steps that are actively harmful
against the current codebase. For current behaviour, use:

| For | Read |
|---|---|
| Everything, first | [`../../CLAUDE.md`](../../CLAUDE.md) — the most reliable file in the repo |
| Pipeline internals | [`../architecture.md`](../architecture.md) |
| Config keys and env vars | [`../configuration.md`](../configuration.md) |
| HTTP API | [`../api.md`](../api.md) |
| Jetson deployment | [`../jetson.md`](../jetson.md) |
| Store/profile design rationale | [`../ADR-multi-corpus-profiles-and-pluggable-store.md`](../ADR-multi-corpus-profiles-and-pluggable-store.md) |

---

## Completed implementation plans

Both describe work that shipped. Kept for the design rationale and the phase
breakdown; the `.claude/agents/*.md` definitions still cite them as the spec for
the effort they drove.

### `BACKEND_API_PLAN.md` — implemented, merged 2026-07-18 (`8fa0b71`)
The FastAPI backend it specifies is live under `src/rag/api/`. Divergences from
the plan as built:

- The package now also contains `routes/answer.py`, `jobs.py`, and `token.py`,
  none of which are in the plan's package listing.
- The plan suggests reusing logic from `pipeline_status.py` for `/status`. That
  was **considered and rejected** during implementation — see
  `BACKEND_API_ORCHESTRATION.md`. Do not re-attempt it on the plan's authority.
- A `queued` job status is specified but was never made reachable; jobs go
  straight to `running` (`src/rag/api/jobs.py`).

### `BACKEND_API_ORCHESTRATION.md` — spent 2026-07-18
The subagent orchestration script for the above. Its "working-tree handling"
section describes a dirty tree from that day, and it claims `pyjwt` is absent
from both lockfiles — it is now pinned in both.

---

## Historical reports (2026-06-05)

These four predate the `src/` layout, the package, the git history, and three
whole subsystems. They describe an MVP of two loose scripts (`index_obsidian.py`,
`query_obsidian.py`) that no longer exist. Read them as history only.

Specific claims that are now **false**, flagged because they are stated as
present-tense invariants:

- **`RAG Project Fix Summary.md`** asserts the chunk ID is
  `SHA-256(path, section_index, chunk_index, chunk[:80])` and "unchanged", in a
  list titled *What Was Not Changed*. It is a hash of the **full chunk text**
  (`src/rag/chunking.py`) — which is exactly what makes incremental indexing
  possible. The same file says the collection is deleted and recreated on every
  run; it is never wiped.
- **`RAG Project Review Report.md`** repeats the delete-and-recreate claim, says
  incremental indexing "is not possible without a redesign" (the redesign
  shipped), reports 6,269 chunks, claims there is no `pyproject.toml`/`Makefile`/
  git repo, targets Python 3.12 (the project targets 3.10 for JetPack), and lists
  the Google Drive vault copy under "What Works" — that copy is now explicitly
  forbidden as a source (see `CLAUDE.md`).
- **`RAG Project Fix Summary - Cryptography Dependency.md`** carries no date at
  all. Its recovery steps say `rm -rf chroma_db` and reindex: that would discard
  ~202k chunks and force a full re-embed (hours on the Jetson) for no reason, and
  it contradicts the never-wipe design. The `cryptography>=3.1` conclusion is
  still correct; the file paths and commands are not.
- **`Frontmatter Fix Report.md`** audits the *vault* repo, not this codebase. Its
  26-file "needs manual review" list was never closed here, its scan count
  (1,639 files) is less than half the current vault, and the audit script it
  references lived in `/tmp` and is gone — so the report cannot be reproduced or
  verified as written.
