# MCP search server

`personal-rag` exposes its semantic retrieval as a local [Model Context Protocol](https://modelcontextprotocol.io/) tool. AI assistants like Claude Desktop, Kiro, and Cursor can search your Obsidian vault and PDF library directly from their chat interface.

The first increment is intentionally minimal:

- stdio transport only (local process, no HTTP yet)
- one tool: `search`
- inputs: nonblank `query` and optional `n_results` (default 8, range 1-50)
- returns complete `query.search()` records with full passage text and metadata

Filters, resources, answer generation, indexing triggers, and remote HTTP transport are not exposed yet.

## Quickstart (5 minutes)

### 1. Install the project and populate your index

```bash
cd /path/to/personal-rag
make install
make index
```

If you already have a populated index, skip `make index`.

### 2. Verify the MCP server starts

```bash
.venv/bin/rag-mcp
```

A working server prints **nothing** and waits on stdin. That silence is correct - it is waiting for an MCP client to connect. Press `Ctrl-C` to stop it.

If you see a traceback instead, check the [Troubleshooting](#troubleshooting) section below.

### 3. Find your absolute paths

MCP hosts launch servers as child processes from their own working directory, so relative paths break. You need absolute paths:

```bash
# The executable
echo "$(pwd)/.venv/bin/rag-mcp"

# The config profile (use whichever profile you want)
echo "$(pwd)/config.personal.yaml"

# The vector store
echo "$(pwd)/chroma_db"
```

Copy these values for the next step.

### 4. Configure your MCP client

Pick your host and paste the configuration below, replacing the placeholder paths with the absolute paths from step 3.

#### Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/personal-rag/.venv/bin/rag-mcp",
      "args": [],
      "env": {
        "RAG_CONFIG_PATH": "/absolute/path/to/personal-rag/config.personal.yaml",
        "RAG_INDEX_PATH": "/absolute/path/to/personal-rag/chroma_db"
      }
    }
  }
}
```

After saving, **fully quit** Claude Desktop (not just close the window) and reopen it.

#### Kiro

Add to `.kiro/settings/mcp.json` (workspace-level) or `~/.kiro/settings/mcp.json` (user-level):

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/personal-rag/.venv/bin/rag-mcp",
      "args": [],
      "env": {
        "RAG_CONFIG_PATH": "/absolute/path/to/personal-rag/config.personal.yaml",
        "RAG_INDEX_PATH": "/absolute/path/to/personal-rag/chroma_db"
      },
      "disabled": false
    }
  }
}
```

Kiro reconnects MCP servers automatically after config changes. You can also reconnect manually from the MCP Server view in the sidebar.

#### Cursor

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "personal-rag": {
      "command": "/absolute/path/to/personal-rag/.venv/bin/rag-mcp",
      "args": [],
      "env": {
        "RAG_CONFIG_PATH": "/absolute/path/to/personal-rag/config.personal.yaml",
        "RAG_INDEX_PATH": "/absolute/path/to/personal-rag/chroma_db"
      }
    }
  }
}
```

#### VS Code (Copilot Agent mode)

Create `.vscode/mcp.json` in your project root:

```json
{
  "servers": {
    "personal-rag": {
      "type": "stdio",
      "command": "/absolute/path/to/personal-rag/.venv/bin/rag-mcp",
      "args": [],
      "env": {
        "RAG_CONFIG_PATH": "/absolute/path/to/personal-rag/config.personal.yaml",
        "RAG_INDEX_PATH": "/absolute/path/to/personal-rag/chroma_db"
      }
    }
  }
}
```

Requires VS Code 1.99+ with the GitHub Copilot extension in Agent mode.

#### Any other stdio host

Configure a stdio-based MCP server with:
- **command**: the absolute path to `.venv/bin/rag-mcp`
- **args**: empty
- **env**: set `RAG_CONFIG_PATH` and `RAG_INDEX_PATH` as shown above

### 5. Test it

Ask your AI assistant a question about something in your vault:

> What do I know about Kubernetes?

The assistant should call the `search` tool and incorporate results from your knowledge base into its answer.

## Verifying the connection

### Check the tool is advertised

In Claude Desktop, look for the hammer icon showing available tools. You should see `search` listed under `personal-rag`.

In Kiro, check the MCP Server view in the sidebar - it should show `personal-rag` as connected with one tool.

In Cursor, check Settings > MCP - the server should show as connected.

### Manual smoke test

You can test the server works end-to-end without a host by running:

```bash
make mcp
```

This starts `rag-mcp` over stdio. It will sit and wait (no output is correct). Press `Ctrl-C` to stop.

For protocol-level testing, use the official MCP Inspector:

```bash
uv run --with "mcp[cli]" mcp dev src/rag/mcp/server.py
```

This opens an interactive web UI where you can call `search` directly and see the raw results.

### Check server logs

The MCP server logs to the same files as `rag-query`:
- Text log: `./logs/rag.log` (or the path set by `RAG_LOG_PATH`)
- SQLite log: `./logs/rag.sqlite` (or the path set by `RAG_LOG_DB_PATH`)

Look for `MCP startup: loading embedding model` and `MCP startup complete` to confirm initialization succeeded.

## Troubleshooting

### Server exits immediately with a traceback

**ModuleNotFoundError: No module named 'mcp'**

The MCP SDK is not installed. Reinstall:

```bash
cd /path/to/personal-rag
uv pip install -r requirements.txt
uv pip install -e . --no-deps
```

**ModuleNotFoundError: No module named 'rag'**

The package is not installed in editable mode:

```bash
uv pip install -e . --no-deps
```

**FileNotFoundError or "collection not found"**

The index path cannot be resolved. Either:
- Set `RAG_INDEX_PATH` to the absolute path of your `chroma_db` directory, or
- Run the MCP server from the project root so relative paths resolve correctly

### Host says "server disconnected" or shows no tools

**Most common cause: relative paths.** The host starts the server from its own working directory, not yours. Use absolute paths for `command` and all `env` values.

**Second most common: stale config.** Fully quit and restart the host after editing its config. Claude Desktop in particular requires a full quit (not just closing the window).

**Third: the executable is not found.** Verify the path exists:

```bash
ls -la /absolute/path/to/personal-rag/.venv/bin/rag-mcp
```

If it does not exist, reinstall:

```bash
cd /path/to/personal-rag
uv pip install -e . --no-deps
```

### Server starts but search returns no results

- Verify your index is populated: `make query Q="test"` should return results
- Check that `RAG_CONFIG_PATH` points to the correct profile
- Check that `RAG_INDEX_PATH` points to the same `chroma_db` the indexer wrote to

### "Internal server error" on tool calls

Check the log files for the underlying exception:

```bash
tail -50 logs/rag.log
```

Common causes:
- The embedding model is not cached locally (first run downloads ~80 MB from HuggingFace)
- Insufficient memory on Jetson (the model + store + reranker share 8 GB)

### Claude Desktop specific

Claude Desktop keeps per-server logs at:
- macOS: `~/Library/Logs/Claude/mcp-server-personal-rag.log`
- Windows: `%APPDATA%\Claude\logs\mcp-server-personal-rag.log`

The main connection log is `mcp.log` in the same directory.

### Windows specific

Use backslashes and the `.exe` extension:

```json
{
  "command": "C:\\path\\to\\personal-rag\\.venv\\Scripts\\rag-mcp.exe"
}
```

## Configuration reference

### Environment variables

| Variable | Purpose |
|----------|---------|
| `RAG_CONFIG_PATH` | Select the active config profile (default: `./config.yaml`) |
| `RAG_INDEX_PATH` | Override the vector store directory |
| `RAG_VAULT_PATH` | Override the vault path in the selected profile |
| `RAG_LOG_PATH` | Override the text log file location |
| `RAG_LOG_DB_PATH` | Override the SQLite log location |

All other `RAG_*` overrides documented in [configuration.md](configuration.md) also work.

### Profile behavior

The active profile's `rerank_default` setting is honored:
- `config.personal.yaml`: reranking **off** (measured to reduce recall@5 on this corpus)
- `config.logmanager.yaml`: reranking **on**

Clients cannot override reranking per-call in this minimal tool. If you want reranking on/off, select the appropriate profile via `RAG_CONFIG_PATH`.

### Tool schema

The `search` tool exposes exactly two parameters:

| Parameter | Type | Required | Default | Constraints |
|-----------|------|----------|---------|-------------|
| `query` | string | yes | - | non-blank after whitespace trimming |
| `n_results` | integer | no | 8 | strict integer, 1-50 |

Unknown parameters are rejected. Non-integer values for `n_results` (booleans, floats, strings) are rejected.

### Response format

Each result is a complete retrieval record:

```json
{
  "document": "Full passage text, never truncated...",
  "metadata": {
    "title": "Kubernetes Notes",
    "path": "Knowledge/DevOps/Kubernetes.md",
    "domain": "DevOps",
    "subdomain": "Container Orchestration",
    "type": "Knowledge",
    "source": "vault",
    "status": "processed",
    "confidence": "high",
    "tags": "kubernetes, containers, devops",
    "heading": "Scheduling"
  },
  "distance": 0.1842,
  "rank": 1,
  "rerank_score": 4.75
}
```

`rerank_score` is present only when reranking was applied. `distance` may be `null` for lexical-only hybrid hits (not currently exposed via MCP).

## Privacy and data retention

### What leaves the device

The MCP adapter itself makes **no outbound network calls**. Retrieval is fully local.

However, the MCP **host** (Claude Desktop, Kiro, Cursor) receives every returned passage and its metadata. Depending on the host's configuration, that data may then be sent to a model provider (Anthropic, OpenAI, etc.) as tool-result context. Review your host's data-retention and model-provider settings before exposing a sensitive corpus.

### What is logged locally

MCP searches flow through `query.search()`, which logs the raw query text at INFO level. With the default logging setup, queries are retained in:

- A rotating text log (default: `./logs/rag.log`)
- A SQLite structured log (default: `./logs/rag.sqlite`)

Paths resolve from `RAG_LOG_PATH` / `RAG_LOG_DB_PATH`, then profile config, then defaults. Use absolute paths in your MCP environment variables when the host's working directory differs from the project root.

Rotate or protect these files as appropriate for your use case.

## Architecture

```
MCP Host (Claude Desktop / Kiro / Cursor)
    │
    │  stdio (JSON-RPC over stdin/stdout)
    │
    ▼
rag-mcp process
    │
    │  lifespan: loads config, model, store once
    │
    ▼
query.search()  ←  the single retrieval seam
    │
    ▼
ChromaStore (src/rag/store/)
```

The MCP server is a thin transport adapter. It never reimplements retrieval, never imports `chromadb` directly, and shares the exact same code path as `rag-query`, `POST /query`, and `make eval`.

## Future plans

Not yet implemented, planned for later increments:

- Metadata filters (`domain`, `tags`, `status`, etc.) as optional tool parameters
- `answer` tool wrapping the generation layer
- `status` tool reporting index health
- Streamable HTTP transport for remote/Jetson access
- Resources exposing the document catalog for browsing
