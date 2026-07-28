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

## egress-audit

Verify nothing has crept into the **indexing or retrieval** path that leaves the
machine. Scope note: this was originally a cloud-*storage* audit; it is now an
egress audit, because `src/rag/generation/` deliberately calls provider APIs and the
old greps would have reported "clean" while it did so.

Last cloud-storage run 2026-07-13: clean (no `google.cloud`/`gsutil`/`boto3`/
`storage.googleapis.com` anywhere in `src/`, `tests/`, `docs/`, `Makefile`,
`Dockerfile*`; only inert legacy `gcs_path` id-namespace strings).

```bash
# 1. cloud storage — expect zero hits
grep -rn "google\.cloud\|google-cloud-storage\|gsutil\|storage\.googleapis\|boto3\|from google\|import google" src/ --include="*.py"
grep -rn "gs://" src/ --include="*.py"

# 2. any outbound HTTP — expect hits ONLY under src/rag/generation/
grep -rn "httpx\|requests\.\|urllib\|aiohttp\|https://" src/rag/ --include="*.py"
```

The second grep is the one that matters now. `src/rag/generation/` is the single
sanctioned egress point (Anthropic/OpenAI, off by default, key from the env). A hit
anywhere else in `src/rag/` — especially under `store/`, `extractors/`, `indexing.py`
or `query.py` — is a finding.

Also spot-check `config.yaml`'s `extractor.books_path`/`resources_path`/
`pdf_sources` still point at local disk, not a cloud-synced folder (Google Drive,
Dropbox, etc.) — this drifted once already, see `CLAUDE.md` gotchas.

## rag-quality-review

Independent assessment of retrieval quality (not code style): is the embedding
model, chunking strategy, and retrieval design actually good for this corpus, given
the Jetson's 8 GB memory constraint. Full report + prioritized roadmap:
`Templates/RAG Quality Review Report.md` in the vault repo. Condensed version lives
in this repo's `CLAUDE.md` → "Known Limitations & Improvement Roadmap."

**Roadmap state as of 2026-07-28:** items 1, 3, 4, 5, 7 done; item 2 resolved as
rejected (both stronger embedding models net-regressed on this corpus); **item 6
(BM25/lexical hybrid) is the only one left**, and it is specced in
`docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md` rather than being loose work.

So the useful trigger for re-running this review is no longer "after items 1-6" —
it is **whenever `make eval` moves materially**, or after any change to the embedding
model, chunking parameters, or the store backend. The eval harness now exists
(item 3), so measure; don't eyeball a few queries:

```bash
make eval                      # profile default
make eval ARGS=--rerank        # compare both modes on the same index
make eval ARGS=--no-rerank
```

Snapshots go in `tests/eval/`. Note `tests/eval/baseline.json` is from 2026-07-20
(205,476 chunks) and predates the current corpus — compare against
`post_update_norerank.json` (202,132) instead, or refresh the baseline.

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
make jetson-index      # build/update the ChromaDB collection
```

**Guarded, but not fully:** if *every* source reports 0 files while the collection
already holds chunks, `indexer.main()` raises `RuntimeError` and prunes nothing —
the error names the config keys and env overrides to check. This is the guard added
after the 2026-07-15 incident, in which exactly this situation pruned 172,557
chunks to zero. You do not need to Ctrl+C to beat the prune step.

What the guard does **not** cover:

- A **partially** broken mount. One source empty while others are fine is
  indistinguishable from a real deletion, and those chunks *will* be pruned. Read the
  per-source file counts in the startup log.
- An **already-empty** index. A fresh profile pointed at a bad path won't trip the
  guard; it will "succeed" and create an empty collection.

If you do need to serve from the Jetson, `make jetson-serve` runs the API container.
Budget memory for it: the server keeps the embedder resident (plus the cross-encoder
once anything reranks) while a reindex subprocess runs alongside it in the same 8 GB.

Full details: `docs/jetson.md`.
