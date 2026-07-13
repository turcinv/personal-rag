# Jetson Orin Nano Super Setup

## Hardware

- **Device:** NVIDIA Jetson Orin Nano Super
- **JetPack:** 6.2
- **CUDA:** 12.6
- **Python:** 3.10 (not 3.12 — JetPack 6.2 ships 3.10)
- **RAM:** 8 GB unified (CPU + GPU share the same pool)

## Local install

PyTorch must come from the Jetson AI Lab index — standard PyPI wheels are x86-only and will not install on aarch64:

```bash
# 1. Install PyTorch first
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 2. Install other deps
pip install -r requirements-jetson.txt

# 3. Install the package entry points
pip install -e . --no-deps
```

## Docker

Build and run **on the Jetson itself**. The PyTorch wheels are aarch64-only and cannot be installed on x86.

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed on the host.

```bash
# First-time build (slow — ~1.5 GB PyTorch layer, cached after first run)
make build-jetson

# Index
make jetson-index

# Query
make jetson-query Q="What do I know about K3s?"

# Smoke tests
make jetson-test
```

Or directly:

```bash
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.indexer
docker compose -f docker-compose.jetson.yml run --rm rag python -m rag.query "your question"
```

The Jetson Compose file sets `runtime: nvidia`, `NVIDIA_VISIBLE_DEVICES=all`, and `NVIDIA_DRIVER_CAPABILITIES=compute,utility` for full GPU access inside the container.

## Data on the Jetson

The compose file mounts host paths from a `.env` in the project root **on the Jetson**.
Only two datasets are required — book/resource PDFs are optional because their content
ships in the pre-extracted JSON (PDF sources are just a fallback for files the
extraction pipeline hasn't covered yet):

| Mount | `.env` variable | Required? | How to get it onto the Jetson |
|---|---|---|---|
| Vault (Markdown notes) | `RAG_VAULT_PATH` | ✅ | `git clone` the Career Knowledge Base repo (or `rsync -a` from the workstation) |
| Pre-extracted JSON | `RAG_JSON_PATH` | ✅ | `rsync -a ~/Documents/knowledge-base-index/indexed/ jetson:~/data/indexed/` |
| Books PDFs | `RAG_PDF_BOOKS_PATH` | optional | only needed to live-parse *new* PDFs not yet in the JSON |
| Resources PDFs | `RAG_PDF_RESOURCES_PATH` | optional | same fallback role |

```bash
# .env on the Jetson (paths are examples)
RAG_VAULT_PATH=/home/turcinv/data/career-knowledge-base
RAG_JSON_PATH=/home/turcinv/data/indexed
# Optional fallback mounts — omit to skip live PDF parsing entirely:
# RAG_PDF_BOOKS_PATH=/home/turcinv/data/books
# RAG_PDF_RESOURCES_PATH=/home/turcinv/data/resources
```

Unset variables default to `/tmp` in the compose file, which simply yields an empty
source. Re-running `make jetson-index` after a vault sync is incremental: chunk IDs
are content-hashed, so unchanged notes are skipped and chunks from deleted/edited
notes are pruned automatically — no need to wipe the `chroma` volume.

## Memory budget

With 8 GB unified RAM shared between CPU and GPU, keep these config values:

| Setting | Value | Reason |
|---|---|---|
| `embedding_batch_size` | `16` | Limits GPU memory per encode call |
| `markdown_workers` | `1` | Sequential MD extraction; avoids parallel RAM spikes |
| `pdf_workers` | `1` | Sequential PDF extraction |

The streaming indexer never accumulates all chunks globally — peak RAM is bounded to one file's chunks at a time.

## Models in the HF cache (offline)

Models are fetched from Hugging Face at first use into the `hf-cache` Docker volume, then reused. Two are needed:

| Model | ~Size | Fetched by | Config key |
|---|---|---|---|
| `all-MiniLM-L6-v2` (embedder) | ~90 MB | first `make jetson-index` | `embedding_model` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) | ~80 MB | first `make jetson-query` / `jetson-eval` (rerank is on by default) | `reranker_model` |

The Jetson needs network access the **first** time each model is used; after that the cache serves them offline. For a fully air-gapped box, pre-populate the `hf-cache` volume (or run one online query with `--no-rerank` omitted so the reranker downloads). Both fit the 8 GB budget alongside the bi-encoder. Disable reranking with `rag-query --no-rerank` / `make eval ARGS=--no-rerank` if you want to skip loading the cross-encoder entirely.

## Why not `encode_multi_process`

Jetson uses NvSCI IPC instead of CUDA IPC. Cross-process CUDA tensor sharing fails on Jetson. The indexer uses single-process GPU encoding only.

## ChromaDB on aarch64

ChromaDB `0.6.3` publishes `manylinux_2_17_aarch64` wheels — no special handling needed.

## libcudss missing at runtime

`torch 2.8+` from `pypi.jetson-ai-lab.io/jp6/cu126` links against `libcudss.so.0` (CUDA Direct Sparse Solver). This library is **not** bundled in `l4t-jetpack:r36.4.0`, **not** in Ubuntu Ports, and **not** in the NVIDIA CUDA sbsa apt repo (that repo is for server ARM64 / GH200, not Jetson).

Symptom: `ImportError: libcudss.so.0: cannot open shared object file: No such file or directory` on the first `import torch`.

**Fix (already in Dockerfile.jetson):** install via NVIDIA's Jetson/Tegra-specific local `.deb` installer:

```dockerfile
RUN wget -q https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb \
    && dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb \
    && cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/ \
    && apt-get update && apt-get install -y cudss
```

To install cuDSS directly on the Jetson host (outside Docker):

```bash
wget https://developer.download.nvidia.com/compute/cudss/0.7.1/local_installers/cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo dpkg -i cudss-local-tegra-repo-ubuntu2204-0.7.1_0.7.1-1_arm64.deb
sudo cp /var/cudss-local-tegra-repo-ubuntu2204-0.7.1/cudss-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update && sudo apt-get install -y cudss
```

**Alternative:** pin torch to `==2.7.*` in the Dockerfile — that version does not link against libcudss at all.
