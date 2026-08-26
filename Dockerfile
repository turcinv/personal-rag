# personal-rag — x86 / macOS (CPU; add --gpus all for NVIDIA GPU)
#
# Build from the project root:
#   docker build -f docker/Dockerfile -t personal-rag:latest .
#   make build

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Dependency layer first — cached across source-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY config.yaml .env.example ./

# Regular (non-editable) install: the package is built into a wheel and copied
# into site-packages, so the deployed image does not depend on the source tree
# staying in place. --no-deps because requirements.txt already pinned everything.
RUN pip install --no-cache-dir . --no-deps

ENV HF_HOME=/data/hf-cache
ENV TRANSFORMERS_CACHE=/data/hf-cache
ENV RAG_INDEX_PATH=/data/chroma

# Keep stdout pure for the MCP stdio handshake (rag-mcp): HF/torch chatter must
# go to stderr, never stdout, or it corrupts the JSON-RPC stream.
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV TOKENIZERS_PARALLELISM=false

# Bake the embedding model into the image (CPU) so a per-session `docker run` has
# no cold-start network pull and works offline. Lands in HF_HOME=/data/hf-cache,
# a different subpath than the /data/chroma mount, so it is never shadowed.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

CMD ["python", "-m", "rag.query", "--help"]
