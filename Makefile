.PHONY: help install index query pipeline-status sync-to-jetson test test-unit build build-jetson \
        docker-index docker-query docker-test \
        jetson-pipeline-status jetson-full-pipeline \
        jetson-index jetson-query jetson-test \
        extract enrich build-index build-notes build-sqlite build-vault-index \
        dup-detect link-mocs search analyze \
        docker-extract docker-enrich docker-build-index docker-build-notes \
        docker-build-sqlite docker-build-vault-index docker-dup-detect docker-link-mocs \
        jetson-extract jetson-enrich jetson-build-index jetson-build-notes \
        jetson-build-sqlite jetson-build-vault-index jetson-dup-detect jetson-link-mocs

PYTHON := .venv/bin/python
Q      ?=
K      ?=

# Jetson sync — set JETSON_HOST and JETSON_OUTPUT_PATH in .env or on the command line.
# Example: make sync-to-jetson JETSON_HOST=turcinv@gpu-01
-include .env
export
JETSON_HOST        ?= gpu-01
JETSON_OUTPUT_PATH ?= ~/knowledge-base-index

help:
	@echo ""
	@echo "Setup:"
	@echo "  make install              create venv and install package"
	@echo ""
	@echo "RAG (local):"
	@echo "  make index                reindex vault + PDFs into ChromaDB"
	@echo "  make query Q=\"...\"        semantic query"
	@echo "  make pipeline-status      check all extraction pipeline outputs"
	@echo "  make sync-to-jetson       rsync all extraction outputs to Jetson (set JETSON_HOST)"
	@echo "  make test-unit            offline pytest unit suite"
	@echo "  make test [K=keyword]     retrieval smoke tests (needs an index)"
	@echo ""
	@echo "Extractor pipeline (local) — run in order for a full pipeline run:"
	@echo "  make analyze              pre-flight survey of books/resources dirs"
	@echo "  make extract              extract text from PDFs/EPUBs (Books + Resources)"
	@echo "  make enrich               enrich inventory metadata from embedded fields"
	@echo "  make build-index          join inventory + text into indexed/*.json"
	@echo "  make build-notes          generate Obsidian Resource Notes"
	@echo "  make build-sqlite         build FTS5 SQLite database"
	@echo "  make build-vault-index    index vault Knowledge/ notes into JSONL"
	@echo "  make dup-detect           near-duplicate detection report"
	@echo "  make link-mocs            inject resource backlinks into Topic MOCs"
	@echo "  make search Q=\"...\"        CLI FTS search over resources.db"
	@echo ""
	@echo "Docker x86:"
	@echo "  make build                build personal-rag:latest"
	@echo "  make docker-index / docker-query Q=\"...\""
	@echo "  make docker-extract / docker-enrich / docker-build-index ..."
	@echo ""
	@echo "Docker Jetson (run on Jetson):"
	@echo "  make build-jetson         build personal-rag:jetson"
	@echo "  make jetson-pipeline-status     check all extraction pipeline outputs"
	@echo "  make jetson-full-pipeline       extract + enrich + build + index (all steps)"
	@echo "  make jetson-index               reindex into the ChromaDB collection"
	@echo "  make jetson-query Q=\"...\"       semantic query"
	@echo "  make jetson-extract / jetson-enrich / jetson-build-index ..."
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	uv venv .venv
	uv pip install -r requirements.txt
	uv pip install -e . --no-deps

# ── Local: RAG ────────────────────────────────────────────────────────────────

index:
	.venv/bin/rag-index

query:
	.venv/bin/rag-query $(Q)

pipeline-status:
	-.venv/bin/rag-pipeline-status

# Sync all extraction outputs (text_output_*, indexed/, resources.db) from macOS → Jetson.
# Set JETSON_HOST in .env or pass on the command line: make sync-to-jetson JETSON_HOST=gpu-01
sync-to-jetson:
	@echo "Syncing knowledge-base-index → $(JETSON_HOST):$(JETSON_OUTPUT_PATH)"
	rsync -avz --progress \
		$(shell python3 -c "import yaml,os; c=yaml.safe_load(open('config.yaml')); print(os.path.expanduser(c.get('extractor',{}).get('output_path','~/Documents/knowledge-base-index')))") \
		$(JETSON_HOST):$(JETSON_OUTPUT_PATH)
	@echo "Done. Run 'make build-jetson && make jetson-index' on the Jetson."

test:
	$(PYTHON) tests/test_queries.py $(K)

test-unit:
	$(PYTHON) -m pytest tests/ -q

# ── Local: Extractor pipeline ─────────────────────────────────────────────────
# Paths are taken from config.yaml extractor: section.
# Use shell expansion to read them from the config if needed, or just pass
# the args directly since the scripts accept CLI arguments.

CATALOG    := $(shell $(PYTHON) -c "import yaml,os; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(e.get('catalog_path',''))" 2>/dev/null)
OUTPUT     := $(shell $(PYTHON) -c "import yaml,os; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(os.path.expanduser(e.get('output_path','~/Documents/knowledge-base-index')))" 2>/dev/null)
BOOKS      := $(shell $(PYTHON) -c "import yaml; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(e.get('books_path',''))" 2>/dev/null)
RESOURCES  := $(shell $(PYTHON) -c "import yaml; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(e.get('resources_path',''))" 2>/dev/null)
NOTES_OUT  := $(shell $(PYTHON) -c "import yaml; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(e.get('obsidian_notes_path',''))" 2>/dev/null)
MOCS       := $(shell $(PYTHON) -c "import yaml; c=yaml.safe_load(open('config.yaml')); e=c.get('extractor',{}); print(e.get('mocs_path',''))" 2>/dev/null)
VAULT      := $(shell $(PYTHON) -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c.get('vault_path',''))" 2>/dev/null)

analyze:
	.venv/bin/rag-analyze "$(BOOKS)"
	.venv/bin/rag-analyze "$(RESOURCES)"

extract:
	.venv/bin/rag-extract "$(BOOKS)"
	.venv/bin/rag-extract "$(RESOURCES)"

enrich:
	.venv/bin/rag-enrich \
	    --inventory "$(CATALOG)/resource_inventory.jsonl" \
	    --source-dir "$(BOOKS)" --source-dir "$(RESOURCES)" \
	    --text-dir "$(BOOKS)/text_output" --text-dir "$(RESOURCES)/text_output" \
	    --out "$(CATALOG)/resource_inventory_enriched.jsonl"

build-index:
	.venv/bin/rag-build-index \
	    --inventory "$(CATALOG)/resource_inventory_enriched.jsonl" \
	    --text-dir "$(BOOKS)/text_output" --text-dir "$(RESOURCES)/text_output" \
	    --out "$(OUTPUT)/indexed"

build-notes:
	.venv/bin/rag-build-notes \
	    --inventory "$(CATALOG)/resource_inventory_enriched.jsonl" \
	    --text-dir "$(BOOKS)/text_output" --text-dir "$(RESOURCES)/text_output" \
	    --generated-dir "$(MOCS)" \
	    --out "$(NOTES_OUT)"

build-sqlite:
	.venv/bin/rag-build-sqlite \
	    --jsonl "$(OUTPUT)/indexed/index_documents.jsonl" \
	    --jsonl "$(OUTPUT)/indexed/vault_documents.jsonl" \
	    --db    "$(OUTPUT)/resources.db"

build-vault-index:
	.venv/bin/rag-build-vault-index \
	    --vault "$(VAULT)" \
	    --out   "$(OUTPUT)/indexed"

dup-detect:
	.venv/bin/rag-dup-detect \
	    --inventory "$(CATALOG)/resource_inventory_enriched.jsonl" \
	    --text-dir "$(BOOKS)/text_output" --text-dir "$(RESOURCES)/text_output" \
	    --out "$(MOCS)/Content Duplicate Candidates.md"

link-mocs:
	.venv/bin/rag-link-mocs \
	    --inventory "$(CATALOG)/resource_inventory_enriched.jsonl" \
	    --generated-dir "$(MOCS)"

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

docker-extract:
	docker compose run --rm rag rag-extract /books
	docker compose run --rm rag rag-extract /resources

docker-enrich:
	docker compose run --rm rag rag-enrich \
	    --inventory /catalog/resource_inventory.jsonl \
	    --source-dir /books --source-dir /resources \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out /catalog/resource_inventory_enriched.jsonl

docker-build-index:
	docker compose run --rm rag rag-build-index \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out /extractor-out/indexed

docker-build-notes:
	docker compose run --rm rag rag-build-notes \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --generated-dir /mocs \
	    --out "/mocs/Resource Notes"

docker-build-sqlite:
	docker compose run --rm rag rag-build-sqlite \
	    --jsonl /extractor-out/indexed/index_documents.jsonl \
	    --jsonl /extractor-out/indexed/vault_documents.jsonl \
	    --db    /extractor-out/resources.db

docker-build-vault-index:
	docker compose run --rm rag rag-build-vault-index \
	    --vault /vault \
	    --out   /extractor-out/indexed

docker-dup-detect:
	docker compose run --rm rag rag-dup-detect \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out "/mocs/Content Duplicate Candidates.md"

docker-link-mocs:
	docker compose run --rm rag rag-link-mocs \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --generated-dir /mocs

# ── Docker Jetson ─────────────────────────────────────────────────────────────

build-jetson:
	docker build -f Dockerfile.jetson -t personal-rag:jetson .

jetson-pipeline-status:
	-docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.pipeline_status

# Full pipeline: extraction → enrichment → build steps → index.
# Requires PDF source mounts (RAG_PDF_BOOKS_PATH / RAG_PDF_RESOURCES_PATH) to be set.
jetson-full-pipeline: jetson-extract jetson-enrich jetson-build-index jetson-build-notes \
                      jetson-build-sqlite jetson-build-vault-index jetson-link-mocs \
                      jetson-index

jetson-index:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer

jetson-query:
	docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query $(Q)

jetson-test:
	docker compose -f docker-compose.jetson.yml run --rm rag python tests/test_queries.py $(K)

jetson-extract:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-extract /books
	docker compose -f docker-compose.jetson.yml run --rm rag rag-extract /resources

jetson-enrich:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-enrich \
	    --inventory /catalog/resource_inventory.jsonl \
	    --source-dir /books --source-dir /resources \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out /catalog/resource_inventory_enriched.jsonl

jetson-build-index:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-build-index \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out /extractor-out/indexed

jetson-build-notes:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-build-notes \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --generated-dir /mocs \
	    --out "/mocs/Resource Notes"

jetson-build-sqlite:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-build-sqlite \
	    --jsonl /extractor-out/indexed/index_documents.jsonl \
	    --jsonl /extractor-out/indexed/vault_documents.jsonl \
	    --db    /extractor-out/resources.db

jetson-build-vault-index:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-build-vault-index \
	    --vault /vault \
	    --out   /extractor-out/indexed

jetson-dup-detect:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-dup-detect \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --text-dir /books/text_output --text-dir /resources/text_output \
	    --out "/mocs/Content Duplicate Candidates.md"

jetson-link-mocs:
	docker compose -f docker-compose.jetson.yml run --rm rag rag-link-mocs \
	    --inventory /catalog/resource_inventory_enriched.jsonl \
	    --generated-dir /mocs
