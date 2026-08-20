"""Offline tests for the stdio MCP search adapter.

Most tests use the SDK's in-memory client with all expensive runtime
dependencies replaced by fakes. One framing test uses a short-lived local
subprocess with the same fake runtime. No model, index, GPU, project data, or
network access is used.
"""

import json
import sys

import pytest
from mcp import Client, StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError
from mcp_types import INVALID_PARAMS

from rag.mcp import server


FULL_RECORD = {
    "document": "Complete passage text that must never be truncated.",
    "metadata": {
        "title": "Kubernetes Notes",
        "path": "Knowledge/DevOps/Kubernetes.md",
        "domain": "DevOps",
        "heading": "Scheduling",
    },
    "distance": 0.125,
    "rank": 1,
    "rerank_score": 4.75,
}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fake_runtime(monkeypatch):
    config = {"embedding_model": "fake-model", "rerank_default": True}
    model = object()
    store = object()
    counters = {
        "load_config": 0,
        "setup_logging": 0,
        "get_model": 0,
        "open_store": 0,
    }
    search_calls = []

    def fake_load_config():
        counters["load_config"] += 1
        return config

    def fake_setup_logging(loaded_config, console):
        counters["setup_logging"] += 1
        assert loaded_config is config
        assert console is False

    def fake_get_model(model_name):
        counters["get_model"] += 1
        assert model_name == "fake-model"
        return model

    def fake_open_store(loaded_config, **kwargs):
        counters["open_store"] += 1
        assert loaded_config is config
        assert kwargs["model"] is model
        return store

    def fake_search(query, n_results=8, **kwargs):
        search_calls.append(
            {"query": query, "n_results": n_results, **kwargs}
        )
        return [FULL_RECORD]

    monkeypatch.setattr(server, "load_config", fake_load_config)
    monkeypatch.setattr(server, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(server.rag_query, "get_model", fake_get_model)
    monkeypatch.setattr(server.rag_query, "open_store", fake_open_store)
    monkeypatch.setattr(server.rag_query, "search", fake_search)

    return {
        "config": config,
        "model": model,
        "store": store,
        "counters": counters,
        "search_calls": search_calls,
    }


@pytest.fixture
async def client(fake_runtime):
    async with Client(server.mcp, raise_exceptions=True) as connected:
        yield connected


def test_server_constructs_without_starting_runtime():
    """Importing the module constructs the server but does not enter lifespan."""
    assert server.mcp.name == "personal-rag"


@pytest.mark.anyio
async def test_advertises_only_minimal_search_tool(client):
    listed = await client.list_tools()

    assert [tool.name for tool in listed.tools] == ["search"]
    schema = listed.tools[0].input_schema
    assert set(schema["properties"]) == {"query", "n_results"}
    assert schema["required"] == ["query"]
    assert schema["properties"]["query"]["minLength"] == 1
    assert schema["properties"]["n_results"]["minimum"] == 1
    assert schema["properties"]["n_results"]["maximum"] == 50
    assert schema["properties"]["n_results"]["default"] == 8


@pytest.mark.anyio
async def test_search_forwards_state_defaults_and_full_records(client, fake_runtime):
    default_result = await client.call_tool(
        "search", {"query": "  What do I know about Kubernetes?  "}
    )
    explicit_result = await client.call_tool(
        "search", {"query": "Kubernetes scheduling", "n_results": 12}
    )

    assert default_result.is_error is False
    assert default_result.structured_content == {"result": [FULL_RECORD]}
    assert FULL_RECORD["document"] in default_result.content[0].text
    assert json.loads(default_result.content[0].text) == FULL_RECORD
    assert explicit_result.structured_content == {"result": [FULL_RECORD]}

    calls = fake_runtime["search_calls"]
    assert len(calls) == 2
    assert calls[0]["query"] == "What do I know about Kubernetes?"
    assert calls[0]["n_results"] == 8
    assert calls[1]["query"] == "Kubernetes scheduling"
    assert calls[1]["n_results"] == 12
    for call in calls:
        assert call["config"] is fake_runtime["config"]
        assert call["model"] is fake_runtime["model"]
        assert call["store"] is fake_runtime["store"]
        assert call["rerank"] is True

    # Both calls shared one lifespan initialization.
    assert fake_runtime["counters"] == {
        "load_config": 1,
        "setup_logging": 1,
        "get_model": 1,
        "open_store": 1,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    "arguments",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "valid", "n_results": 0},
        {"query": "valid", "n_results": 51},
        {"query": "valid", "n_results": True},
        {"query": "valid", "n_results": 8.0},
        {"query": "valid", "n_results": "8"},
        {"query": "valid", "n_results": None},
    ],
)
async def test_invalid_inputs_do_not_call_retrieval(fake_runtime, arguments):
    async with Client(server.mcp) as connected:
        result = await connected.call_tool("search", arguments)

    assert result.is_error is True
    assert fake_runtime["search_calls"] == []


@pytest.mark.anyio
async def test_unknown_arguments_are_rejected(fake_runtime):
    async with Client(server.mcp) as connected:
        with pytest.raises(MCPError) as exc_info:
            await connected.call_tool(
                "search", {"query": "valid", "unexpected": "value"}
            )

    assert exc_info.value.code == INVALID_PARAMS
    assert "unexpected" in exc_info.value.message
    assert fake_runtime["search_calls"] == []


def test_main_runs_stdio_transport(monkeypatch):
    transports = []
    monkeypatch.setattr(server.mcp, "run", transports.append)

    server.main()

    assert transports == ["stdio"]


@pytest.mark.anyio
async def test_stdio_subprocess_handshake_and_search(tmp_path):
    """Exercise real JSON-RPC framing without loading project data or models."""
    wrapper = tmp_path / "fake_mcp_server.py"
    wrapper.write_text(
        "\n".join(
            [
                "from rag.mcp import server",
                "config = {'embedding_model': 'fake', 'rerank_default': False}",
                "server.load_config = lambda: config",
                "server.setup_logging = lambda config, console: None",
                "server.rag_query.get_model = lambda name: object()",
                "server.rag_query.open_store = lambda config, **kwargs: object()",
                f"record = {FULL_RECORD!r}",
                "server.rag_query.search = lambda query, n_results=8, **kwargs: [record]",
                "server.main()",
            ]
        ),
        encoding="utf-8",
    )
    process = StdioServerParameters(
        command=sys.executable,
        args=[str(wrapper)],
        cwd=tmp_path,
    )

    async with Client(stdio_client(process), read_timeout_seconds=10) as connected:
        listed = await connected.list_tools()
        result = await connected.call_tool("search", {"query": "Kubernetes"})

    assert [tool.name for tool in listed.tools] == ["search"]
    assert result.is_error is False
    assert result.structured_content == {"result": [FULL_RECORD]}
