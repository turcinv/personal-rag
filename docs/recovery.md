# Backup, restore, and writer coordination

This covers how `personal-rag` protects the vector store against concurrent
writers and how to recover it. It is deliberately short: the index is a
*derived* artifact, so the corpus (vault + PDFs/EPUBs) is the real source of
truth and the ultimate fallback.

## Writer coordination

All processes that mutate the index or a generation-bound sidecar hold a single
**cross-process writer lock**, an advisory `flock` on
`<index_path>.writer.lock` (keyed by the resolved `index_path`, so every writer
that shares one Chroma directory contends on one lock):

- `rag-index` (CLI and the API's spawned reindex subprocess) — held across the
  whole mutating span (generation bump → upserts/deletes → run manifest).
- `rag-build-lexical` — held while it reads the store's records/generation and
  publishes the sidecar, so a concurrent reindex cannot bump the generation
  mid-build.
- `rag-backup create` — held while copying, so a backup is never taken of a
  database another process is writing (the "quiesced" requirement).
- `scripts/drop_collections.py` — held while dropping collections.

A second writer fails fast with the current owner's operation, run id, pid, and
host. Pass `--wait-for-lock SECONDS` to `rag-index` / `rag-build-lexical` to
block instead. A dry-run index (`rag-index --dry-run`) is read-only and takes no
lock.

**Stale locks self-heal.** `flock` is released by the kernel when the holding
process dies, so a crashed writer never leaves a permanently stuck lock; the
next writer reclaims it (and logs that it did). There is no manual "break lock"
step and no PID file to clean up.

## Backup

```bash
make backup DEST=backups/2026-08-20
```

`rag-backup create` acquires the writer lock, then copies into `DEST`:

- `chroma/` — the whole Chroma index dir (vectors **and** provenance, since
  provenance lives in the collection metadata inside `chroma.sqlite3`),
- `lexical/<collection>.db` — the BM25 sidecar, if present,
- `manifests/` — the index-run manifests,
- `config.yaml` — the active config,
- `backup_manifest.json` — restore metadata: chunk count, generation id +
  provenance fingerprint, and a `sha256` of every copied Chroma file.

It then verifies the backup in place (checksums + chunk count + provenance).

## Restore and drill

```bash
# Restore to a fresh index path and verify counts/provenance/query:
make restore DIR=backups/2026-08-20 DEST=chroma_restored

# Non-destructive drill: restore to a temp path, verify, and clean up:
make restore-drill DIR=backups/2026-08-20
```

Restore refuses a non-empty destination and never writes the live index unless
you explicitly point `DEST` at it. Verification opens the restored copy through
the normal store seam and checks chunk count and generation/provenance
fingerprint against the manifest, plus a model-free representative query (a
fixed vector of the recorded dimension) to prove the index answers.

## RPO / RTO

- **RPO (data loss window):** whatever has changed in the corpus since the last
  `make index`. The index has no independent data — every chunk is reproducible
  from the vault/PDF sources — so a lost index costs *time*, not *data*.
- **RTO (time to recover):**
  - *From a backup* (fast path): a file copy + verify — seconds to a few
    minutes depending on index size. Use this to roll back a bad reindex or move
    the index between machines.
  - *From raw sources* (fallback / rebuild): re-run the extractor pipeline (if
    books/resources changed) and `rag-index`, then `rag-build-lexical`. This is
    the authoritative rebuild and the recovery of last resort; it takes
    considerably longer (full re-embed) but needs no backup.

Prefer restoring from a backup for speed and to preserve the exact generation
id; fall back to a raw-source rebuild when no trustworthy backup exists or when
the embedding model/chunking changed (which requires a fresh index anyway).
