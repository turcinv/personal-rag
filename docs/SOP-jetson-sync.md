# SOP: Sync code + data to the Jetson and reindex

Standard procedure for pushing the current state of `personal-rag` (code), the
vault, and the extraction outputs from the macOS workstation to the Jetson Orin
Nano (`gpu-01`), then rebuilding the index there.

**When to run:** after adding/editing vault notes, adding new books (after the
extraction pipeline has run), or changing `personal-rag` code that affects
indexing or serving.

**Time:** ~5 min for an incremental sync + reindex; a full re-embed (only after
chunking/model changes) takes considerably longer on the Jetson.

---

## Prerequisites (one-time)

- SSH alias `jetson` configured in `~/.ssh/config` → `turcinv@gpu-01`.
- Target directories exist on the Jetson (`~/personal-rag`,
  `~/personal_knowledge/...`, `~/knowledge-base-index`) and are owned by
  `turcinv` — if a Docker run ever created one as root:
  `sudo chown -R turcinv:turcinv ~/knowledge-base-index`.
- `~/personal-rag/.env` **on the Jetson** has the four indexer vars set (see
  step 4). This file is never synced from the workstation.

## Step 1 — Sync the code

> If `~/personal-rag` on the Jetson is a git clone, prefer
> `ssh jetson 'cd ~/personal-rag && git pull'` and skip the rsync. Pick one
> method and stick with it — rsync onto a git checkout dirties its working tree.

```bash
rsync -avz --progress \
  --exclude '.venv/' \
  --exclude 'chroma_db/' \
  --exclude 'lexical_index/' \
  --exclude 'logs/' \
  --exclude '.env' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  ~/Documents/personal-rag/ jetson:~/personal-rag/
```

Do **not** add `--delete`. Do **not** sync `.env` or `chroma_db/` — the Jetson
keeps its own.

## Step 2 — Sync the vault

```bash
rsync -avz --progress --exclude '.git/' \
  ~/Documents/personal_knowledge/Career\ Knowledge\ Base/ \
  "jetson:~/personal_knowledge/Career Knowledge Base/"
```

(Or `git pull` inside the checkout on the Jetson if it is one — same rule as
step 1.)

⚠️ Never sync into or point the indexer at `~/mindmap/Career Knowledge Base/`
on either machine — that is the stale Google Drive copy with divergent history.

## Step 3 — Sync the extraction outputs

```bash
# From the workstation repo root; JETSON_HOST can also live in the local .env
make sync-to-jetson JETSON_HOST=jetson
# Preview the transfer plan without sending anything:
make sync-to-jetson JETSON_HOST=jetson ARGS="--dry-run"
```

`sync-to-jetson` now runs `rag-sync-generation`, which transfers exactly one
**validated, immutable** artifact generation rather than the whole mutable tree.
It:

1. validates the active generation's manifest (schema, checksums, counts, SQLite
   integrity) before planning anything,
2. copies `generations/<id>/` first (immutable — safe to resend),
3. copies the compatibility alias symlinks (`indexed`, `resources.db`,
   `text_output_books`, `text_output_resources`),
4. copies the `current` pointer **last** — the activation.

Because the pointer moves last and no `--delete` is used, the Jetson always sees
either the previous complete generation or the next one, and old generations
remain for rollback. The **lexical index is never synced** — its rows are keyed
to the workstation's vector generation ID; rebuild it on the Jetson with
`make build-lexical` after `make jetson-index`.

**Failure: "Permission denied" on the receiver** → the target dir is
root-owned from an old Docker run. Fix on the Jetson:
`sudo chown -R turcinv:turcinv ~/knowledge-base-index`, then retry.

## Step 4 — Verify `.env` and mounts on the Jetson ⚠️ MANDATORY

This is the step whose omission wiped the collection on 2026-07-15 (172,557
chunks → 0): an unset path var makes Compose mount the host's empty `/tmp`,
which the indexer reads as "all sources deleted".

```bash
ssh jetson
cd ~/personal-rag
grep -E 'RAG_VAULT_PATH|RAG_JSON_PATH' .env
```

Both must be set to the real paths, e.g.:

```
RAG_VAULT_PATH=/home/turcinv/personal_knowledge/Career Knowledge Base
RAG_JSON_PATH=/home/turcinv/knowledge-base-index/indexed
```

Then confirm what Compose actually resolves:

```bash
docker compose -f docker-compose.jetson.yml config | grep -A2 volumes
```

**Stop here if any mount resolves to `/tmp`.** Source-scoped reconciliation now
preserves chunks owned by a missing, degraded, unreadable, or unexpectedly empty
source while healthy sources reconcile independently; the all-sources-zero guard is
an additional backstop. Do not use either prune override to bypass an unexplained
mount problem.

## Step 5 — Rebuild the image (only if code changed)

The image copies the source, so any code sync requires a rebuild. Skip this
step for data-only syncs.

```bash
make build-jetson
```

## Step 6 — Reindex

```bash
make jetson-index
```

Watch the per-source file counts in the startup log — expected order of
magnitude: ~3,000 vault markdown files, ~260 JSON docs. **A source reporting 0
files means a broken mount — abort and go back to step 4.** The run is
incremental (content-hashed chunk IDs): unchanged chunks are skipped, stale
ones pruned; never wipe the `chroma` volume manually.

## Step 7 — Verify

```bash
make jetson-query Q="What do I know about K3s?"
make jetson-eval          # optional: recall@5/@10 + MRR vs tests/eval/baseline.json
```

Also check for silently dropped notes (broken YAML frontmatter):

```bash
grep -c "SKIP .*mapping values" logs/rag.log
```

Any hit = a note with zero chunks; fix by quoting the offending frontmatter
value and reindexing.

## Step 8 — Restart the API (if serving)

```bash
make jetson-serve         # requires RAG_API_JWT_SECRET in the Jetson .env
```

The server loads the model/store once at startup, so it must be restarted to
pick up a new image or reindexed collection.

---

## Quick reference (routine update, no code changes)

```bash
# workstation
rsync -avz --exclude '.git/' ~/Documents/personal_knowledge/Career\ Knowledge\ Base/ "jetson:~/personal_knowledge/Career Knowledge Base/"
make sync-to-jetson JETSON_HOST=jetson   # transfers the active validated generation, pointer last

# jetson
cd ~/personal-rag
docker compose -f docker-compose.jetson.yml config | grep -A2 volumes   # mounts sane?
make jetson-index
```
