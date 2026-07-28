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

Re-running `make jetson-index` after a vault sync is incremental: chunk IDs are
content-hashed, so unchanged notes are skipped and chunks from deleted/edited notes
are pruned automatically — no need to wipe the `chroma` volume.

> **An unset variable is not a harmless no-op.** Compose resolves it to
> `${VAR:-/tmp}` and mounts the host's `/tmp`, so the container sees a real but empty
> directory — indistinguishable, to the indexer, from a source whose files were all
> deleted. Combined with incremental pruning, that is what emptied the collection on
> 2026-07-15 (172,557 chunks → 0).
>
> `indexer.main()` now raises `RuntimeError` and prunes nothing when **every** source
> reports 0 files while the index holds chunks, so that exact failure can no longer
> wipe anything. It does **not** protect a *partially* broken mount: if the vault
> mounts but `RAG_JSON_PATH` doesn't, the book/resource chunks look legitimately
> deleted and will be pruned. Verify the per-source counts in the startup log, and
> if in doubt check what Compose actually resolved:
>
> ```bash
> docker compose -f docker-compose.jetson.yml config | grep -A2 volumes
> ```

## Memory budget

With 8 GB unified RAM shared between CPU and GPU, keep these config values:

| Setting | Value | Reason |
|---|---|---|
| `embedding_batch_size` | `16` | Limits GPU memory per encode call |
| `markdown_workers` | `1` | Sequential MD extraction; avoids parallel RAM spikes |
| `pdf_workers` | `1` | Sequential PDF extraction |

The streaming indexer never accumulates all chunks globally — peak RAM is bounded to one file's chunks at a time.

**If you also run the API server**, budget for it separately: `rag-serve` holds the
embedder (~90 MB) resident for the life of the process, plus the cross-encoder
(~80 MB) once anything has requested a rerank, plus the Python/uvicorn baseline.
That is the whole point of the server — the bots don't pay a cold start — but it
means a reindex triggered while the server is up runs *alongside* those resident
models. Indexing runs as a **subprocess**, so it has its own address space and a
failure cannot take the server's models down with it; the two still share the same
8 GB. For a large reindex on a memory-tight box, stop the server first.

## Running the API server

```bash
make jetson-serve    # docker compose -f docker-compose.jetson.yml up api
```

Needs `RAG_API_JWT_SECRET` in `.env` (≥32 bytes) — protected routes return 500
without it. Reached over Tailscale, internal only, no in-app TLS. Full endpoint
reference and the Python client for the bots: [api.md](api.md).

> Config profiles do not work in Docker: the image copies only `config.yaml` and
> Compose does not pass `RAG_CONFIG_PATH`. The Jetson container therefore always
> serves the default profile.

## Models in the HF cache (offline)

Models are fetched from Hugging Face at first use into the `hf-cache` Docker volume, then reused. Two are needed:

| Model | ~Size | Fetched by | Config key |
|---|---|---|---|
| `all-MiniLM-L6-v2` (embedder) | ~90 MB | first `make jetson-index` | `embedding_model` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker) | ~80 MB | first query that **explicitly** reranks — see below | `reranker_model` |

The Jetson needs network access the **first** time each model is used; after that the cache serves them offline. Both fit the 8 GB budget alongside the bi-encoder.

> **Air-gap prep: you must pass `--rerank` explicitly.**
> Since 2026-07-27 the personal/Jetson profile sets `rerank_default: false`, so a
> plain `make jetson-query` **never loads the cross-encoder** and therefore never
> downloads it. Omitting `--no-rerank` is no longer enough — that was true only while
> reranking defaulted on.
>
> ```bash
> make jetson-query Q="warm the reranker cache" ARGS=--rerank
> ```
>
> Do this once while the box still has network, or pre-populate the `hf-cache`
> volume by hand. Otherwise the first `--rerank` query on an air-gapped Jetson fails
> at model load.

To skip the cross-encoder entirely, just don't ask for it: the profile default is already off. Use `--rerank` per call when you want it (and see CLAUDE.md roadmap item 5 for why it is off — it measurably loses recall@5 on this corpus).

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
