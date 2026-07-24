"""Anthropic Messages API backend for the generation layer.

Calls the Messages API directly over ``httpx`` (no ``anthropic`` SDK dependency
— one fewer heavy pin, and the request/response shape is small and stable). The
API key is read from the configured env var at construction; a request never
carries it. Grounding + citation rules come from the shared prompt builder in
:mod:`rag.generation.base`, so this and the OpenAI backend answer identically.
"""

import logging
from typing import Optional

from .base import (
    DEFAULT_SYSTEM_PROMPT,
    AnswerResult,
    GenerationError,
    build_prompt,
)

logger = logging.getLogger("rag")

DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicGenerator:
    """Generate grounded answers via Anthropic's Messages API.

    Bound to one ``model`` + API key. ``client`` may be injected (an
    ``httpx.Client``) so tests can supply a ``MockTransport``; in production it
    is created lazily with the configured ``timeout``.
    """

    provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout: float = 60.0,
        base_url: str = DEFAULT_BASE_URL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client=None,
    ):
        self._api_key = api_key
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")
        self._system_prompt = system_prompt
        self._client = client

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def generate(
        self,
        question: str,
        contexts: list,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AnswerResult:
        payload = {
            "model": self._model,
            "max_tokens": max_tokens or self._max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
            "system": self._system_prompt,
            "messages": [
                {"role": "user", "content": build_prompt(question, contexts)}
            ],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            resp = self._get_client().post(
                f"{self._base_url}/v1/messages", json=payload, headers=headers
            )
        except Exception as exc:  # transport/timeout — never leak as a 500
            raise GenerationError(f"Anthropic request failed: {exc}") from exc

        if resp.status_code >= 400:
            # Surface the provider's error body (trimmed) but never the api key.
            raise GenerationError(
                f"Anthropic API returned {resp.status_code}: {resp.text[:500]}"
            )

        try:
            body = resp.json()
            # content is a list of typed blocks; concatenate the text blocks.
            text = "".join(
                block.get("text", "")
                for block in body.get("content", [])
                if block.get("type") == "text"
            ).strip()
        except Exception as exc:
            raise GenerationError(f"Anthropic response was not parseable: {exc}") from exc

        if not text:
            raise GenerationError("Anthropic response contained no text content.")

        return AnswerResult(
            answer=text,
            model=body.get("model", self._model),
            usage=body.get("usage"),
            stop_reason=body.get("stop_reason"),
            raw=body,
        )
