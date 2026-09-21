.PHONY: help install install-dev check lint typecheck test-coverage package-check config-doctor index build-lexical query eval stats pipeline-status sync-to-jetson test test-unit build build-jetson \
        serve mcp mcp-docker pipeline docker-pipeline docker-serve jetson-serve \
        docker-index docker-query docker-test \
        jetson-pipeline-status jetson-full-pipeline \
        jetson-index jetson-query jetson-eval jetson-stats jetson-test \
        extract enrich build-index build-notes build-books-index build-sqlite build-vault-index \
        dup-detect link-mocs search analyze \
        backup backup-verify restore restore-drill \
        docker-extract docker-enrich docker-build-index docker-build-notes docker-build-books-index \
        docker-build-sqlite docker-build-vault-index docker-dup-detect docker-link-mocs \
        jetson-extract jetson-enrich jetson-build-index jetson-build-notes jetson-build-books-index \
        jetson-build-sqlite jetson-build-vault-index jetson-dup-detect jetson-link-mocs

PYTHON := .venv/bin/python
Q      ?=
K      ?=
ARGS   ?=

# Jetson sync — set JETSON_HOST and JETSON_OUTPUT_PATH in .env or on the command line.
# Example: make sync-to-jetson JETSON_HOST=turcinv@gpu-01
-include .env
export
JETSON_HOST        ?= gpu-01
JETSON_OUTPUT_PATH ?= ~/knowledge-base-index

help:
	@echo ""
	@echo "Setup:"
	@echo "  make install              create venv and install runtime package"
	@echo "  make install-dev          install runtime plus pinned contributor tools"
	@echo "  make check                lint, type-check, coverage-tested unit suite, wheel smoke test"
	@echo ""
	@echo "RAG (local):"
	@echo "  make index                reindex vault + PDFs into ChromaDB"
	@echo "  make build-lexical        build BM25 lexical index for hybrid (after index)"
	@echo "  make query Q=\"...\"        semantic query"
	@echo "  make config-doctor        validate config and report resolved source paths"
	@echo "  make pipeline-status      check all extraction pipeline outputs"
	@echo "  make eval [ARGS=...]      recall@k / MRR eval over golden_queries.jsonl"
	@echo "  make stats [ARGS=...]     index statistics (chunk counts, source breakdown)"
	@echo "  make serve                run the HTTP API locally (rag-serve, port 8000)"
	@echo "  make mcp                  run the local stdio MCP search server"
	@echo "  make sync-to-jetson       transfer the active validated generation to Jetson (set JETSON_HOST)"
	@echo "  make backup DEST=...      quiesced backup of index + sidecars (holds writer lock)"
	@echo "  make restore-drill DIR=.. restore a backup to a temp path and verify it"
	@echo "  make test-unit            offline pytest unit suite"
	@echo "  make test-coverage        unit suite with branch-coverage ratchet"
	@echo "  make lint                 Ruff correctness checks"
	@echo "  make typecheck            mypy static type check (lenient, src only)"
	@echo "  make package-check        build wheel and smoke-test entry points"
	@echo "  make test [K=keyword]     retrieval smoke report (needs an index)"
	@echo ""
	@echo "Extractor pipeline (local):"
	@echo "  make pipeline [ARGS=...]  run the dependency-ordered full artifact pipeline"
	@echo "  make pipeline ARGS=\"--dry-run\"  print stage status and resolved commands"
	@echo "  make analyze              pre-flight survey of books/resources dirs"
	@echo "  make extract              extract text from PDFs/EPUBs (Books + Resources)"
	@echo "  make enrich               enrich inventory metadata from embedded fields"
	@echo "  make build-index          join inventory + text into indexed/*.json"
	@echo "  make build-notes          generate Obsidian Resource Notes"
	@echo "  make build-books-index    regenerate the Books Index aggregate note"
	@echo "  make build-sqlite         build FTS5 SQLite database"
	@echo "  make build-vault-index    index vault Knowledge/ notes into JSONL"
	@echo "  make dup-detect           near-duplicate detection report"
	@echo "  make link-mocs            inject resource backlinks into Topic MOCs"
	@echo "  make search Q=\"...\"        CLI FTS search over resources.db"
	@echo ""
	@echo "Docker x86:"
	@echo "  make build                build personal-rag:latest"
	@echo "  make mcp-docker           build the image + print the Docker MCP wrapper path"
	@echo "  make docker-index / docker-query Q=\"...\""
	@echo "  make docker-serve         run the HTTP API container (port 8000)"
	@echo "  make docker-pipeline      run the config-driven artifact pipeline"
	@echo "  make docker-extract / docker-enrich / docker-build-index ..."
	@echo ""
	@echo "Docker Jetson (run on Jetson):"
	@echo "  make build-jetson         build personal-rag:jetson"
	@echo "  make jetson-pipeline-status     check all extraction pipeline outputs"
	@echo "  make jetson-full-pipeline       extract + enrich + build + index (all steps)"
	@echo "  make jetson-index               reindex into the ChromaDB collection"
	@echo "  make jetson-query Q=\"...\"       semantic query"
	@echo "  make jetson-eval [ARGS=...]     recall@k / MRR eval over golden_queries.jsonl"
	@echo "  make jetson-stats [ARGS=...]    index statistics on Jetson"
	@echo "  make jetson-serve               run the HTTP API container (port 8000)"
	@echo "  make jetson-extract / jetson-enrich / jetson-build-index ..."
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	uv venv .venv
	uv pip install -r requirements.txt
	uv pip install -e . --no-deps

install-dev: install
	uv pip install -e ".[dev]"

check: lint typecheck test-coverage package-check

lint:
	$(PYTHON) -m ruff check src tests scripts

typecheck:
	$(PYTHON) -m mypy

test-coverage:
	$(PYTHON) -m pytest tests/ -q --cov=rag --cov=extractor --cov-branch

# Build into a temporary directory, install the wheel under a temporary prefix,
# then invoke representative generated console scripts against that wheel.
package-check:
	@set -eu; \
	tmpdir=$$(mktemp -d); \
	trap 'rm -rf "$$tmpdir"' EXIT; \
	mkdir -p "$$tmpdir/project"; \
	cp pyproject.toml "$$tmpdir/project/"; \
	cp -R src "$$tmpdir/project/"; \
	$(PYTHON) -m build --no-isolation --wheel --outdir "$$tmpdir/dist" "$$tmpdir/project" >/dev/null; \
	uv pip install --quiet --no-deps --prefix "$$tmpdir/prefix" "$$tmpdir"/dist/*.whl; \
	site="lib/python$$($(PYTHON) -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-index" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-query" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-stats" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-config" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-pipeline" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-extract" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-sync-generation" --help >/dev/null; \
	PYTHONPATH="$$tmpdir/prefix/$$site" "$$tmpdir/prefix/bin/rag-backup" --help >/dev/null; \
	echo "Wheel and representative console scripts verified."

# ── Local: RAG ────────────────────────────────────────────────────────────────

index:
	.venv/bin/rag-index

# Build the BM25 lexical index (SQLite FTS5) from the existing ChromaDB collection,
# for client-side hybrid retrieval (rag-query --hybrid / make eval ARGS=--hybrid).
# Run after `make index`; rebuilds ./lexical_index/<collection>.db (gitignored).
build-lexical:
	.venv/bin/rag-build-lexical

query:
	.venv/bin/rag-query $(Q)

config-doctor:
	.venv/bin/rag-config $(ARGS)

pipeline-status:
	-.venv/bin/rag-pipeline-status

# Recall@k / MRR eval over tests/eval/golden_queries.jsonl (needs a populated index).
# Record a baseline:  make eval ARGS="--label baseline --out tests/eval/baseline.json"
# A/B a variant:      make eval ARGS="--label dense --no-rerank"
eval:
	$(PYTHON) scripts/eval_recall.py $(ARGS)

stats:
	.venv/bin/rag-stats $(ARGS)

# Quiesced backup / verified restore of the index + sidecars (holds the writer lock).
#   make backup DEST=backups/2026-08-20
#   make backup-verify DIR=backups/2026-08-20
#   make restore DIR=backups/2026-08-20 DEST=chroma_restored
#   make restore-drill DIR=backups/2026-08-20   # restore to a temp path, verify, clean up
backup:
	.venv/bin/rag-backup create $(DEST) $(ARGS)

backup-verify:
	.venv/bin/rag-backup verify $(DIR)

restore:
	.venv/bin/rag-backup restore $(DIR) $(DEST) $(ARGS)

restore-drill:
	.venv/bin/rag-backup drill $(DIR)

# Run the HTTP API locally. Reads RAG_API_HOST/RAG_API_PORT/RAG_API_JWT_SECRET
# from the environment / .env. Loads model + collection + reranker once.
serve:
	.venv/bin/rag-serve

# Run the local MCP search server over stdio. The client owns stdin/stdout;
# configure an MCP host with the absolute path to .venv/bin/rag-mcp.
mcp:
	.venv/bin/rag-mcp

# Build the image (bakes the embedding model) and print where to register the
# Docker MCP wrapper. See docs/mcp.md → "Docker option" for the wrapper contents.
mcp-docker:
	docker build -t personal-rag:latest .
	@echo "Built personal-rag:latest."
	@echo "Register the wrapper at ~/tools/personal-rag-mcp.sh — see docs/mcp.md → 'Docker option'."

# Transfer the active, validated artifact generation (and its compatibility
# aliases) from macOS → Jetson, flipping the remote `current` pointer last so a
# consumer always sees a complete generation. The lexical index and Chroma store
# are never included. Set JETSON_HOST / JETSON_OUTPUT_PATH in .env or on the CLI.
# Preview the plan without transferring: make sync-to-jetson ARGS="--dry-run"
sync-to-jetson:
	.venv/bin/rag-sync-generation --host $(JETSON_HOST) --remote-path $(JETSON_OUTPUT_PATH) $(ARGS)

test:
	$(PYTHON) tests/test_queries.py $(K)

test-unit:
	$(PYTHON) -m pytest tests/ -q

# ── Local: Extractor pipeline ─────────────────────────────────────────────────
# The Python coordinator owns config resolution, stage dependencies, and paths.

pipeline:
	.venv/bin/rag-pipeline $(ARGS)

analyze:
	.venv/bin/rag-pipeline --stage analyze --no-deps $(ARGS)

extract:
	.venv/bin/rag-pipeline --stage extract --no-deps $(ARGS)

enrich:
	.venv/bin/rag-pipeline --stage enrich --no-deps $(ARGS)

build-index:
	.venv/bin/rag-pipeline --stage build-index --no-deps $(ARGS)

build-notes:
	.venv/bin/rag-pipeline --stage build-notes --no-deps $(ARGS)

build-books-index:
	.venv/bin/rag-pipeline --stage build-books-index --no-deps $(ARGS)

build-sqlite:
	.venv/bin/rag-pipeline --stage build-sqlite --no-deps $(ARGS)

build-vault-index:
	.venv/bin/rag-pipeline --stage build-vault-index --no-deps $(ARGS)

dup-detect:
	.venv/bin/rag-pipeline --stage dup-detect --no-deps $(ARGS)

link-mocs:
	.venv/bin/rag-pipeline --stage link-mocs --no-deps $(ARGS)

search:
	.venv/bin/rag-search $(Q)

# ── Docker x86 ────────────────────────────────────────────────────────────────

build:
	docker build -t personal-rag:latest .

docker-index:
	docker compose run --rm rag python -m rag.indexer

docker-query:
	docker compose run --rm rag python -m rag.query $(Q)

docker-test:
	docker compose run --rm rag python tests/test_queries.py $(K)

# Long-running HTTP API server (port 8000). Set RAG_API_JWT_SECRET in .env.
docker-serve:
	docker compose up api

docker-pipeline:
	docker compose run --rm rag rag-pipeline $(ARGS)

docker-extract:
	docker compose run --rm rag rag-pipeline --stage extract --no-deps $(ARGS)

docker-enrich:
	docker compose run --rm rag rag-pipeline --stage enrich --no-deps $(ARGS)

docker-build-index:
	docker compose run --rm rag rag-pipeline --stage build-index --no-deps $(ARGS)

docker-build-notes:
	docker compose run --rm rag rag-pipeline --stage build-notes --no-deps $(ARGS)

docker-build-books-index:
	docker compose run --rm rag rag-pipeline --stage build-books-index --no-deps $(ARGS)

docker-build-sqlite:
	docker compose run --rm rag rag-pipeline --stage build-sqlite --no-deps $(ARGS)

docker-build-vault-index:
	docker compose run --rm rag rag-pipeline --stage build-vault-index --no-deps $(ARGS)

docker-dup-detect:
	docker compose run --rm rag rag-pipeline --stage dup-detect --no-deps $(ARGS)

docker-link-mocs:
	docker compose run --rm rag rag-pipeline --stage link-mocs --no-deps $(ARGS)

# ── Docker Jetson ─────────────────────────────────────────────────────────────

build-jetson:
	docker build -f Dockerfile.jetson -t personal-rag:jetson .

jetson-pipeline-status:
	-docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.pipeline_status

# Full pipeline: extraction artifacts followed by vector indexing.
# Requires all source mounts to be set; rag-pipeline resolves paths from config/env.
jetson-full-pipeline:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline $(ARGS)
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer

jetson-index:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer

jetson-query:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query $(Q)

jetson-eval:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.eval $(ARGS)

jetson-stats:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-stats $(ARGS)

# Long-running HTTP API server (port 8000). Set RAG_API_JWT_SECRET in .env.
jetson-serve:
	docker compose -f docker-compose.jetson.yml up api

jetson-test:
	docker compose -f docker-compose.jetson.yml run --rm rag python tests/test_queries.py $(K)

jetson-extract:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage extract --no-deps $(ARGS)

jetson-enrich:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage enrich --no-deps $(ARGS)

jetson-build-index:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage build-index --no-deps $(ARGS)

jetson-build-notes:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage build-notes --no-deps $(ARGS)

jetson-build-books-index:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage build-books-index --no-deps $(ARGS)

jetson-build-sqlite:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage build-sqlite --no-deps $(ARGS)

jetson-build-vault-index:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage build-vault-index --no-deps $(ARGS)

jetson-dup-detect:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage dup-detect --no-deps $(ARGS)

jetson-link-mocs:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-pipeline --stage link-mocs --no-deps $(ARGS)
