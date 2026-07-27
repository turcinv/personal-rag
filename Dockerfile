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

# Editable install must come AFTER src/ exists: this is a src-layout package,
# so setuptools puts egg-info under src/ and errors out if the dir is missing.
RUN pip install --no-cache-dir -e . --no-deps

ENV HF_HOME=/data/hf-cache
ENV TRANSFORMERS_CACHE=/data/hf-cache
ENV RAG_INDEX_PATH=/data/chroma

CMD ["python", "-m", "rag.query", "--help"]
