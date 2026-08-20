"""Stdio MCP server exposing the personal-rag retrieval seam.

The module is import-safe: model loading and store access happen only when the
server lifespan starts. This keeps unit tests offline and lets MCP clients pay
the model/store startup cost once per server process rather than once per call.
"""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS
from pydantic import Field, StringConstraints

from .. import query as rag_query
from ..utils import load_config, setup_logging

logger = logging.getLogger("rag")

QueryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ResultCount = Annotated[
    int,
    Field(strict=True, ge=1, le=50),
]


@dataclass
class AppContext:
    """Heavy retrieval state shared by every MCP tool call."""

    config: dict
    model: Any
    store: Any


@asynccontextmanager
async def lifespan(_server: MCPServer) -> AsyncIterator[AppContext]:
    """Load the active profile, embedding model, and store exactly once."""
    config = load_config()
    # stdout is the stdio protocol channel. Project logs must never use it.
    setup_logging(config, console=False)

    embedding_model = config.get(
        "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
    )
    logger.info("MCP startup: loading embedding model %s", embedding_model)
    model = rag_query.get_model(embedding_model)

    logger.info("MCP startup: opening store")
    store = rag_query.open_store(config, model=model)
    logger.info("MCP startup complete")

    try:
        yield AppContext(config=config, model=model, store=store)
    finally:
        logger.info("MCP shutdown")


async def reject_unknown_search_arguments(ctx, call_next):
    """Reject undeclared search inputs before SDK argument coercion."""
    if ctx.method == "tools/call" and isinstance(ctx.params, Mapping):
        if ctx.params.get("name") == "search":
            arguments = ctx.params.get("arguments") or {}
            if isinstance(arguments, Mapping):
                unknown = set(arguments) - {"query", "n_results"}
                if unknown:
                    names = ", ".join(sorted(str(name) for name in unknown))
                    raise MCPError(
                        code=INVALID_PARAMS,
                        message=f"Unknown search argument(s): {names}",
                    )
    return await call_next(ctx)


mcp = MCPServer(
    "personal-rag",
    lifespan=lifespan,
    middleware=[reject_unknown_search_arguments],
)


@mcp.tool()
def search(
    query: QueryText,
    ctx: Context[AppContext],
    n_results: ResultCount = 8,
) -> list[dict[str, Any]]:
    """Search the local knowledge base for passages relevant to a question.

    Returns complete retrieval records, including each passage's full text,
    metadata, distance, rank, and rerank score when the active profile enables
    reranking. Use a small result count unless more context is necessary.
    """
    state = ctx.request_context.lifespan_context
    # The MCP surface is intentionally query-only: no metadata filters and no
    # hybrid toggle (host clients pass free-text questions). Filtering/hybrid are
    # CLI/eval surfaces. rerank follows the active profile default.
    return rag_query.search(
        query,
        n_results=n_results,
        config=state.config,
        model=state.model,
        store=state.store,
        rerank=rag_query.rerank_default(state.config),
    )


def main() -> None:
    """Run the personal-rag MCP server over the local stdio transport."""
    mcp.run("stdio")


if __name__ == "__main__":
    main()
