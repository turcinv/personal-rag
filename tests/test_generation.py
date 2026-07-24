"""Offline unit tests for the answer-generation layer (rag.generation).

No network: the provider clients are driven with ``httpx.MockTransport`` so the
real Anthropic/OpenAI APIs are never called. Covers the shared prompt builder,
the ``get_generator`` factory, and each backend's request shaping + response
parsing + error handling.
"""

import json

import httpx
import pytest

from rag.generation import (
    AnswerResult,
    GenerationConfigError,
    GenerationError,
    build_prompt,
    format_contexts,
    get_generator,
)
from rag.generation.anthropic_gen import AnthropicGenerator
from rag.generation.openai_gen import OpenAIGenerator

# ── fixtures / helpers ──────────────────────────────────────────────────────────

RECORDS = [
    {
        "document": "Docker layers are cached by instruction.",
        "metadata": {"title": "Docker Caching", "path": "DevOps/docker.md", "domain": "DevOps"},
        "distance": 0.1,
        "rank": 1,
    },
    {
        "document": "Use multi-stage builds to shrink images.",
        "metadata": {"title": "Multi-stage", "path": "DevOps/multistage.md", "domain": "DevOps"},
        "distance": 0.2,
        "rank": 2,
    },
]


def _mock_client(handler):
    """An httpx.Client whose requests are served by ``handler(request)->Response``."""
    return httpx.Client(transport=httpx.MockTransport(handler))


# ── shared prompt builder ─────────────────────────────────────────────────────


def test_format_contexts_numbers_and_headers():
    block = format_contexts(RECORDS)
    assert "[1] Docker Caching — DevOps/docker.md" in block
    assert "[2] Multi-stage — DevOps/multistage.md" in block
    # full chunk text is included, not truncated
    assert "Docker layers are cached by instruction." in block


def test_format_contexts_falls_back_to_path_then_untitled():
    recs = [
        {"document": "x", "metadata": {"path": "only/path.md"}},
        {"document": "y", "metadata": {}},
    ]
    block = format_contexts(recs)
    assert "[1] only/path.md" in block
    assert "[2] untitled" in block


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt("How does docker cache?", RECORDS)
    assert "How does docker cache?" in prompt
    assert "[1] Docker Caching" in prompt
    assert "citing each claim with its [n]" in prompt


# ── get_generator factory ─────────────────────────────────────────────────────


def test_get_generator_no_provider_raises():
    with pytest.raises(GenerationConfigError, match="No generation.provider"):
        get_generator({})


def test_get_generator_unknown_provider_raises():
    with pytest.raises(GenerationConfigError, match="Unknown generation provider"):
        get_generator({"generation": {"provider": "llama.cpp", "model": "m"}})


def test_get_generator_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(GenerationConfigError, match="API key env var"):
        get_generator({"generation": {"provider": "anthropic", "model": "claude-x"}})


def test_get_generator_missing_model_raises(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with pytest.raises(GenerationConfigError, match="generation.model is required"):
        get_generator({"generation": {"provider": "anthropic"}})


def test_get_generator_builds_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    gen = get_generator(
        {"generation": {"provider": "anthropic", "model": "claude-x", "max_tokens": 256}}
    )
    assert isinstance(gen, AnthropicGenerator)
    assert gen.provider == "anthropic"
    assert gen.model == "claude-x"


def test_get_generator_builds_openai_with_custom_key_env(monkeypatch):
    monkeypatch.setenv("MY_KEY", "sk-openai")
    gen = get_generator(
        {"generation": {"provider": "openai", "model": "gpt-x", "api_key_env": "MY_KEY"}}
    )
    assert isinstance(gen, OpenAIGenerator)
    assert gen.provider == "openai"


# ── AnthropicGenerator ────────────────────────────────────────────────────────


def test_anthropic_generate_parses_response():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "model": "claude-x-dated",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 20},
                "content": [{"type": "text", "text": "Docker caches layers [1]."}],
            },
        )

    gen = AnthropicGenerator("sk-test", "claude-x", client=_mock_client(handler))
    result = gen.generate("How does docker cache?", RECORDS, max_tokens=222)

    assert isinstance(result, AnswerResult)
    assert result.answer == "Docker caches layers [1]."
    assert result.model == "claude-x-dated"
    assert result.usage == {"input_tokens": 100, "output_tokens": 20}
    assert result.stop_reason == "end_turn"
    # request shaping
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["x-api-key"] == "sk-test"
    assert captured["headers"]["anthropic-version"]
    sent = json.loads(captured["body"])
    assert sent["max_tokens"] == 222
    assert sent["messages"][0]["role"] == "user"
    assert sent["system"]  # grounding system prompt is sent


def test_anthropic_concatenates_multiple_text_blocks():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "claude-x",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "thinking", "text": "IGNORED"},
                    {"type": "text", "text": "Part two."},
                ],
            },
        )

    gen = AnthropicGenerator("sk-test", "claude-x", client=_mock_client(handler))
    result = gen.generate("q", RECORDS)
    assert result.answer == "Part one. Part two."


def test_anthropic_http_error_raises_generation_error():
    def handler(request):
        return httpx.Response(429, text='{"error": "rate limited"}')

    gen = AnthropicGenerator("sk-test", "claude-x", client=_mock_client(handler))
    with pytest.raises(GenerationError, match="429"):
        gen.generate("q", RECORDS)


def test_anthropic_empty_text_raises():
    def handler(request):
        return httpx.Response(200, json={"model": "claude-x", "content": []})

    gen = AnthropicGenerator("sk-test", "claude-x", client=_mock_client(handler))
    with pytest.raises(GenerationError, match="no text content"):
        gen.generate("q", RECORDS)


def test_anthropic_transport_failure_raises():
    def handler(request):
        raise httpx.ConnectError("boom")

    gen = AnthropicGenerator("sk-test", "claude-x", client=_mock_client(handler))
    with pytest.raises(GenerationError, match="request failed"):
        gen.generate("q", RECORDS)


# ── OpenAIGenerator ───────────────────────────────────────────────────────────


def test_openai_generate_parses_response():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Use multi-stage [2]."},
                    }
                ],
            },
        )

    gen = OpenAIGenerator("sk-openai", "gpt-x", client=_mock_client(handler))
    result = gen.generate("How to shrink images?", RECORDS)

    assert result.answer == "Use multi-stage [2]."
    assert result.model == "gpt-x"
    assert result.stop_reason == "stop"
    assert result.usage["total_tokens"] == 120
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["headers"]["authorization"] == "Bearer sk-openai"
    # system + user turns
    sent = json.loads(captured["body"])
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


def test_openai_http_error_raises():
    def handler(request):
        return httpx.Response(500, text="server error")

    gen = OpenAIGenerator("sk-openai", "gpt-x", client=_mock_client(handler))
    with pytest.raises(GenerationError, match="500"):
        gen.generate("q", RECORDS)


def test_openai_empty_content_raises():
    def handler(request):
        return httpx.Response(
            200, json={"model": "gpt-x", "choices": [{"message": {"content": ""}}]}
        )

    gen = OpenAIGenerator("sk-openai", "gpt-x", client=_mock_client(handler))
    with pytest.raises(GenerationError, match="no message content"):
        gen.generate("q", RECORDS)


def test_openai_custom_base_url():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"model": "local", "choices": [{"message": {"content": "hi"}}]},
        )

    gen = OpenAIGenerator(
        "sk", "local", base_url="http://localhost:11434", client=_mock_client(handler)
    )
    gen.generate("q", RECORDS)
    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
