# MCP search server

`personal-rag` exposes its existing semantic retrieval seam as a local Model
Context Protocol (MCP) tool. The first increment is intentionally small:

- stdio transport only;
- one tool, `search`;
- inputs: nonblank `query` and optional `n_results` (default 8, range 1–50);
- complete `query.search()` records, including full passage text and metadata.

Filters, resources, answer generation, indexing, and remote HTTP transport are
not exposed yet.

## Prerequisites

Install the project and populate the selected profile's index:

```bash
make install
make index
```

The embedding model, and the reranker when the profile enables it, must already
be available in the device's Hugging Face cache for fully offline retrieval.
The MCP adapter itself adds no outbound network calls. It must run on the device
that can read the local index; stdio is cross-platform but is not a remote
transport.

## Privacy and retention

The configured MCP host receives every returned passage and its metadata in
full. Trust that host and review its model-provider/data-retention settings:
Claude Desktop, Kiro, Cursor, or another host may send tool results to a remote
model even though retrieval and the MCP adapter are local. Do not expose a
sensitive corpus to a host whose data boundary is unacceptable.

MCP searches reuse `query.search()`, whose existing info log includes the raw
query text. With the default logging setup, queries are retained in a rotating
text log and a SQLite log. Paths resolve from `RAG_LOG_PATH` and
`RAG_LOG_DB_PATH`, then profile settings, then `./logs/rag.log` and the
corresponding `.sqlite` path. Use absolute environment paths when the MCP host's
working directory is not the project root, and protect or rotate those files as
appropriate.

The executable is:

- macOS/Linux: `<checkout>/.venv/bin/rag-mcp`
- Windows: `<checkout>/.venv/Scripts/rag-mcp.exe`

Run it manually with `make mcp` or the absolute executable path. A healthy stdio
server prints nothing and waits for an MCP client on stdin. Stop it with
`Ctrl-C`. The official SDK explains why MCP hosts launch stdio servers as child
processes and why absolute paths are important in its
[host setup guide](https://py.sdk.modelcontextprotocol.io/get-started/real-host/index.md).

## Profile and path configuration

Desktop hosts usually start child processes from their own working directory
and with a reduced environment. Use absolute paths for the executable and these
environment variables:

```text
RAG_CONFIG_PATH=/absolute/path/to/personal-rag/config.personal.yaml
RAG_INDEX_PATH=/absolute/path/to/personal-rag/chroma_db
```

`RAG_CONFIG_PATH` selects the active profile. `RAG_INDEX_PATH` avoids resolving
a relative `index_path` against the host application's working directory. Other
supported `RAG_*` overrides continue to work as documented in
[configuration.md](configuration.md).

The active profile's `rerank_default` is honored. On the personal profile this
is currently disabled because it reduced measured recall; clients cannot
override it in this minimal tool.

## Claude Desktop

On macOS, edit
`~/Library/Application Support/Claude/claude_desktop_config.json`. On Windows,
edit `%APPDATA%\Claude\claude_desktop_config.json`:

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

Use `.venv\\Scripts\\rag-mcp.exe` for `command` on Windows. Fully quit and
restart Claude Desktop after editing its configuration.

## Kiro

Add the server to the workspace `.kiro/settings/mcp.json` or the user-level
`~/.kiro/settings/mcp.json`:

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

Kiro reconnects MCP servers after configuration changes. You can also reconnect
it from the MCP Server view.

## Cursor and other stdio hosts

Cursor uses the same `mcpServers` command/args shape in `.cursor/mcp.json`:

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

For another MCP host, configure a stdio server with the same absolute command,
no arguments, and environment values.

## Using the tool

Ask the host a knowledge-base question, for example:

> What do I know about Kubernetes?

The host can call:

```json
{
  "query": "What do I know about Kubernetes?",
  "n_results": 8
}
```

Every result includes the full `document`, `metadata`, `distance`, and `rank`,
plus `rerank_score` when reranking was applied. Full records preserve source
context but can consume substantial model context; request fewer results when a
small evidence set is sufficient.

For protocol-level testing, the official MCP SDK documents its
[in-memory client](https://py.sdk.modelcontextprotocol.io/get-started/testing/index.md),
which is also what this project's offline MCP tests use.

_Content from the MCP SDK documentation was rephrased for compliance with
licensing restrictions._
