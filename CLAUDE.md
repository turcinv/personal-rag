# personal-rag

Local retrieval index for an Obsidian knowledge vault + a PDF book/resource library. Embeds content as vector chunks in ChromaDB and retrieves by semantic query. Also contains the full document extraction pipeline (formerly `doc-text-extractor`). **Production target: an NVIDIA Jetson Orin Nano** (runs in the `Dockerfile.jetson` container); also runs on macOS/x86 for development.

**Canonical vault** is the maintained git checkout at `~/Documents/personal_knowledge/Career Knowledge Base/` (the Google Drive copy at `mindmap/Career Knowledge Base/` has a stale, divergent history — do not index it). **Books/resources** are indexed from pre-extracted JSON produced by the `src/extractor/` pipeline (`~/Documents/knowledge-base-index/indexed/`), not by parsing PDFs live.

## What this project does

- Extracts text from PDF/EPUB/Markdown source documents (PyMuPDF + Tesseract OCR)
- Builds a classified, FTS-searchable SQLite index of all books/resources + vault notes
- Generates per-resource Obsidian note stubs and injects backlinks into Topic MOCs
- Indexes vault Markdown notes and document JSON into a local ChromaDB collection
- Retrieves relevant chunks by semantic similarity query
- Fully offline — no cloud APIs used for indexing or retrieval
- Compatible with both macOS/x86 and NVIDIA Jetson (Python 3.10, same codebase)

## Environment

**Always use the local `.venv`, never the system Python or Conda.**

```bash
# First-time setup
make install          # creates .venv and installs package + deps

# RAG indexing and querying
.venv/bin/rag-index
.venv/bin/rag-query "your question"
.venv/bin/rag-query "your question" -n 12    # optional: number of results (default 8)
.venv/bin/rag-query "Kubernetes" --domain DevOps

# Extraction pipeline (run in order for a full pipeline run)
make extract          # extract text from Books + Resources dirs
make enrich           # enrich inventory metadata
make build-index      # join inventory + text into indexed/*.json
make build-notes      # generate Obsidian Resource Notes
make build-sqlite     # build FTS5 SQLite database
make build-vault-index  # index vault Knowledge/ notes into JSONL
make dup-detect       # near-duplicate detection report
make link-mocs        # inject resource backlinks into Topic MOCs
make search Q="..."   # CLI FTS search over resources.db

# Run offline unit tests (pytest). Install dev deps first: uv pip install -e ".[dev]"
make test-unit                      # or: .venv/bin/python -m pytest tests/

# Run retrieval smoke tests (needs a populated index)
.venv/bin/python tests/test_queries.py
.venv/bin/python tests/test_queries.py kubernetes   # filter by keyword

# Recreate environment from scratch
uv venv .venv
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

Do not use bare `python` or `python3` — the Conda base environment will be picked up instead of `.venv`.

## Project layout

```
personal-rag/
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── utils.py          # load_config, logging setup, telemetry suppression, .env
│   │   ├── chunking.py       # heading split, char chunking, stable IDs, wikilinks
│   │   ├── extractors/       # one module per source type + iter_sources() registry
│   │   │   ├── __init__.py    #   Source namedtuple + iter_sources(config)
│   │   │   ├── markdown.py    #   extract_md_file, should_exclude
│   │   │   ├── pdf.py         #   extract_pdf_file, clean_pdf_title
│   │   │   └── json_doc.py    #   extract_json_doc (pre-extracted indexed/*.json)
│   │   ├── indexing.py       # incremental engine: embed/upsert, per-file diff, run_source
│   │   ├── indexer.py        # main() orchestration (MD + PDF + JSON → ChromaDB); entry: rag-index
│   │   └── query.py          # semantic query CLI; entry: rag-query
│   └── extractor/            # document extraction pipeline (merged from doc-text-extractor)
│       ├── __init__.py
│       ├── extract_text.py    # PDF/EPUB/MD → text_output/*.json; entry: rag-extract
│       ├── analyze_files.py   # pre-flight survey of a dir; entry: rag-analyze
│       ├── enrich_metadata.py # enrich inventory from embedded fields + ISBNs; entry: rag-enrich
│       ├── build_index_documents.py  # join inventory + text; entry: rag-build-index
│       ├── build_obsidian_notes.py   # generate Resource Notes; entry: rag-build-notes
│       ├── build_sqlite.py    # FTS5 SQLite database; entry: rag-build-sqlite
│       ├── build_vault_index.py  # vault notes → JSONL; entry: rag-build-vault-index
│       ├── dup_detect.py      # near-duplicate detection; entry: rag-dup-detect
│       ├── link_mocs.py       # inject backlinks into MOCs; entry: rag-link-mocs
│       └── search.py          # CLI FTS search; entry: rag-search
├── tests/
│   ├── test_chunking.py      # unit: chunking/ID helpers
│   ├── test_extractors.py    # unit: markdown/pdf/json extractors (rag package)
│   ├── test_extractor.py     # unit: extractor package (detect_type, merge, shingles, etc.)
│   ├── test_indexing.py      # unit: incremental engine + idempotency (fake model + temp chroma)
│   └── test_queries.py       # 13-query retrieval smoke tests (needs a populated index)
├── .github/workflows/ci.yml  # CI: runs the offline unit suite on push/PR (Python 3.10)
├── Dockerfile                # x86 / macOS container image (python:3.10-slim + tesseract)
├── Dockerfile.jetson         # Jetson JetPack 6.2 container image (build on Jetson)
├── docker-compose.yml        # x86 Compose with volume mounts (RAG + extractor paths)
├── docker-compose.jetson.yml # Jetson Compose (runtime: nvidia)
├── .venv/                    # local virtualenv — never commit
├── chroma_db/                # ChromaDB data — never commit
├── .env                      # path overrides — never commit (see .env.example)
├── .env.example              # template for .env
├── .dockerignore
├── config.yaml               # vault path, pdf + json sources, model, chunk settings, extractor paths
├── pyproject.toml            # package definition and CLI entry points (rag + extractor)
├── Makefile                  # shortcuts: make install/index/query/test/extract/enrich/...
├── requirements-direct.txt   # direct deps only — use to regenerate lockfile
├── requirements.txt          # full pinned lockfile — x86 / macOS (compiled for Python 3.10)
├── requirements-jetson.txt   # pinned deps — Jetson JetPack 6.2 (aarch64, CUDA 12.6)
├── docs/
│   ├── architecture.md       # pipeline walkthrough, chunk IDs, telemetry workaround
│   ├── configuration.md      # full config.yaml reference and env var overrides
│   └── jetson.md             # Jetson install, Docker, memory budget, GPU constraints
├── README.md                 # setup and usage guide
└── CLAUDE.md                 # this file
```

## Key dependencies (pinned)

| Package | Version |
|---|---|
| chromadb | 0.6.3 |
| sentence-transformers | 3.3.1 |
| python-frontmatter | 1.3.0 |
| PyYAML | 6.0.3 |
| pypdf | 6.12.2 |
| PyMuPDF | >=1.24 (PDF/EPUB extraction, OCR page rendering) |
| cryptography | >=3.1 (AES-encrypted PDF support) |

## What gets indexed (and how) — current behavior

- **Vault Markdown** (~2,148 notes): frontmatter → metadata. Before embedding, each note's
  body is stripped of the navigation tail (`# Related Topics` / `## Potential New Notes`) and
  inline `[[wikilink]]` syntax — the link graph still lands in the `wikilinks` metadata field
  (extracted from the full body first). See `chunking.strip_navigation_tail` / `strip_wikilink_syntax`.
- **Books/resources**: from `json_sources` (`indexed/*.json`) — full text + enriched metadata.
- **PDF sources are a fallback only**: any PDF whose `file_name` is covered by a JSON doc is
  **skipped** (dedup), so books index exactly once. Live PDF parsing only handles new files the
  extractor hasn't processed yet. (See `extractors.iter_sources` / `_json_covered_filenames`.)
- **Excluded**: `Resources/Generated` (auto catalog stubs that double-cover JSON), `* MOC.md`
  (navigation-only wikilink indexes, via `exclude_filename_patterns`), plus Templates/Archive/etc.

## ChromaDB state

- Collection: `obsidian_markdown`
- Metadata fields: `path`, `title`, `heading`, `type`, `domain`, `status`, `source`, `confidence`, `tags`, `wikilinks`
- Reindexing is incremental: chunk IDs are SHA-256 of the full chunk content, so unchanged chunks are skipped, changed/new ones embedded, and stale chunks (edited or deleted sources) pruned. The collection is never wiped. **Note:** the nav-tail/wikilink stripping changed chunk text, so the first run after those changes re-embeds the vault and prunes the old chunks (expected one-time churn).

## Deployment (Jetson Orin Nano)

- Run `make build-jetson` then `make jetson-index` **on the Jetson** (aarch64 PyTorch wheels can't build on x86). Compose mounts host paths via `.env`.
- **Two datasets must be synced to the Jetson** (the rest is optional fallback): the vault checkout (`git`/`rsync`) and `~/Documents/knowledge-base-index/indexed/` (`rsync`). PDF mounts (`RAG_PDF_BOOKS_PATH` / `RAG_PDF_RESOURCES_PATH`) are optional.
- **Gotcha:** Docker may create the mounted `indexed/` dir as **root** → rsync fails with "Permission denied". Fix on the Jetson: `sudo chown -R turcinv:turcinv ~/knowledge-base-index`.
- Adding new books: run the extractor pipeline (`make extract → make enrich → make build-index`) then rsync `indexed/` to the Jetson and run `make jetson-index`.
- Full details in [docs/jetson.md](docs/jetson.md) ("Data on the Jetson" section).

## Adding new books (end-to-end)

```bash
# 1. Drop the PDF/EPUB into the Books or Resources Google Drive folder
# 2. Add the classification record to Resources/_catalog/resource_inventory.jsonl
# 3. Run the extraction pipeline
make extract        # extract text
make enrich         # enrich metadata
make build-index    # produce indexed/*.json
make build-notes    # update Resource Notes in the vault
make build-sqlite   # update FTS database
make link-mocs      # update MOC backlinks

# 4. Reindex into ChromaDB
make index          # or: make jetson-index if running on Jetson
```

## Deeper reference

| Document | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Pipeline walkthrough, chunk IDs, ChromaDB state, telemetry workaround |
| [docs/configuration.md](docs/configuration.md) | All `config.yaml` fields, env var overrides, hardware tuning table |
| [docs/jetson.md](docs/jetson.md) | Jetson-specific install, Docker, memory budget, IPC constraints |
