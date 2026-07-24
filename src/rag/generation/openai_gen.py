"""OpenAI Chat Completions backend for the generation layer.

Calls the Chat Completions API directly over ``httpx`` (no ``openai`` SDK
dependency). Because the request/response shape is the de-facto standard, this
same backend also drives OpenAI-compatible servers (vLLM, LM Studio, Ollama's
``/v1`` shim, or Anthropic's OpenAI-compat endpoint) by overriding ``base_url``
— useful if the deployment ever moves off the native cloud API. Grounding +
citation rules come from the shared prompt builder in
:mod:`rag.generation.base`.
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

DEFAULT_BASE_URL = "https://api.openai.com"


class OpenAIGenerator:
    """Generate grounded answers via OpenAI's Chat Completions API.

    Bound to one ``model`` + API key. ``client`` may be injected (an
    ``httpx.Client``) so tests can supply a ``MockTransport``. ``base_url`` can
    point at any OpenAI-compatible server.
    """

    provider = "openai"

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
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": build_prompt(question, contexts)},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

        try:
            resp = self._get_client().post(
                f"{self._base_url}/v1/chat/completions", json=payload, headers=headers
            )
        except Exception as exc:  # transport/timeout — never leak as a 500
            raise GenerationError(f"OpenAI request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise GenerationError(
                f"OpenAI API returned {resp.status_code}: {resp.text[:500]}"
            )

        try:
            body = resp.json()
            choice = (body.get("choices") or [{}])[0]
            text = (choice.get("message") or {}).get("content", "").strip()
            stop_reason = choice.get("finish_reason")
        except Exception as exc:
            raise GenerationError(f"OpenAI response was not parseable: {exc}") from exc

        if not text:
            raise GenerationError("OpenAI response contained no message content.")

        return AnswerResult(
            answer=text,
            model=body.get("model", self._model),
            usage=body.get("usage"),
            stop_reason=stop_reason,
            raw=body,
        )
