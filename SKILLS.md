# SKILLS.md

Recurring workflows for this repo, documented so any agent (or future you) can run
them without re-deriving the steps. Not a formal skill-loader — just a catalog.
Cross-references to `Career Knowledge Base` are to the sibling vault repo this
project indexes; those files are Reference notes under `Templates/`, not code.

## extract-pipeline

Full extraction pipeline for new books/resources, in order (each step depends on
the previous one's output — do not skip `build-vault-index` before `build-sqlite`,
it needs `vault_documents.jsonl` which only that step produces):

```bash
make extract enrich build-index build-notes build-vault-index build-sqlite
make dup-detect link-mocs   # optional, but run after adding several new resources
```

See `CLAUDE.md` → "Adding new books (end-to-end)" for the full walkthrough,
including the catalog-row step (`Resources/_catalog/resource_inventory.jsonl`) that
must happen before `build-index` for brand-new files to actually get indexed —
`build-index` reports "N extracted without inventory match" for anything missing a
catalog row, and those files are silently excluded from the index until one exists.

## gcs-dependency-audit

Verify no live cloud-storage calls have crept back into the extractor/indexer
code. Last run 2026-07-13: clean (no `google.cloud`/`gsutil`/`boto3`/
`storage.googleapis.com` anywhere in `src/`, `tests/`, `docs/`, `Makefile`,
`Dockerfile*`; only inert legacy `gcs_path` id-namespace strings). Re-run
periodically or after any change touching the extractor/catalog code:

```bash
grep -rn "google\.cloud\|google-cloud-storage\|gsutil\|storage\.googleapis\|boto3\|from google\|import google" --include="*.py" src/
grep -rn "gs://" --include="*.py" src/
```

Also spot-check `config.yaml`'s `extractor.books_path`/`resources_path`/
`pdf_sources` still point at local disk, not a cloud-synced folder (Google Drive,
Dropbox, etc.) — this drifted once already, see `CLAUDE.md` gotchas.

## rag-quality-review

Independent assessment of retrieval quality (not code style): is the embedding
model, chunking strategy, and retrieval design actually good for this corpus, given
the Jetson's 8 GB memory constraint. Full report + prioritized roadmap:
`Templates/RAG Quality Review Report.md` in the vault repo. Condensed version lives
in this repo's `CLAUDE.md` → "Known Limitations & Improvement Roadmap." Re-run this
review after implementing roadmap items 1-6 there, to confirm the changes actually
moved the needle (needs the recall@k eval set from roadmap item 3 to be meaningful
— don't just eyeball a few queries).

## jetson-deploy

Sync + build + index on the Jetson. Two datasets required (PDF mounts are optional
fallback): the vault checkout and `~/Documents/knowledge-base-index/indexed/`.

```bash
# from the Mac, after extract-pipeline has produced fresh indexed/*.json:
make sync-to-jetson JETSON_HOST=turcinv@<jetson-tailscale-host>

# on the Jetson, in this repo, with .env set (RAG_VAULT_PATH, RAG_JSON_PATH —
# unset vars silently fall back to mounting /tmp, i.e. an empty source, so always
# check the indexer's startup log for non-zero file counts before letting it run
# to completion):
make build-jetson      # first time only, ~1.5 GB PyTorch layer, cached after
make jetson-index-all  # or make jetson-index for the single default collection
```

**Danger:** if the indexer runs with all sources reporting 0 files (misconfigured
`.env`), its incremental-prune logic treats every existing chunk as "deleted from
source" and starts removing them from ChromaDB. If you see all-zero source counts
at startup, Ctrl+C immediately — do not let it reach the prune step. Fix `.env`
first, verify non-zero counts, then re-run.

Full details: `docs/jetson.md`.
