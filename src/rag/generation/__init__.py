"""Pluggable answer-generation layer (retrieval ≠ chatbot).

The generation layer sits above ``rag.query.search``: it takes a question plus
the retrieved chunks and asks an LLM for a grounded, cited answer. It is
orthogonal to the ``RetrievalStore`` backend — see
``docs/ADR-multi-corpus-profiles-and-pluggable-store.md`` (the ADR calls this
out as a separate track added *above* ``search()``, identical for both stores).

``Generator`` is the Protocol every backend implements; ``AnthropicGenerator``
and ``OpenAIGenerator`` are the cloud backends. ``get_generator()`` is the
factory the API lifespan uses, keyed on the ``generation.provider`` config.
"""

import os

from .base import (
    AnswerResult,
    Generator,
    GenerationConfigError,
    GenerationError,
    build_prompt,
    format_contexts,
)

__all__ = [
    "Generator",
    "AnswerResult",
    "GenerationError",
    "GenerationConfigError",
    "build_prompt",
    "format_contexts",
    "get_generator",
]

# provider name -> (module attr path, default model, default api-key env var)
_PROVIDERS = {
    "anthropic": ("anthropic_gen", "AnthropicGenerator", "ANTHROPIC_API_KEY"),
    "openai": ("openai_gen", "OpenAIGenerator", "OPENAI_API_KEY"),
}


def get_generator(config: dict):
    """Instantiate the ``Generator`` backend named in ``config['generation']``.

    Reads the ``generation`` block::

        generation:
          provider: anthropic | openai
          model: <model id>
          max_tokens: 1024
          temperature: 0.0
          timeout: 60
          api_key_env: ANTHROPIC_API_KEY   # optional; provider default otherwise
          base_url: <optional override>

    The API key is pulled from the environment (``api_key_env`` or the provider
    default), NOT from config — keys never live in a committed file. Raises
    :class:`GenerationConfigError` for an unknown provider (a config typo) or a
    missing API key (an expected "generation not wired here" state that the API
    lifespan catches to keep ``/query`` working while ``/answer`` returns 503).
    """
    gen_cfg = config.get("generation") or {}
    provider = gen_cfg.get("provider")
    if not provider:
        raise GenerationConfigError(
            "No generation.provider configured. Set generation.provider to "
            "'anthropic' or 'openai' (and export the API key) to enable /answer."
        )
    if provider not in _PROVIDERS:
        raise GenerationConfigError(
            f"Unknown generation provider {provider!r} — expected one of "
            f"{sorted(_PROVIDERS)}."
        )

    module_name, class_name, default_key_env = _PROVIDERS[provider]
    key_env = gen_cfg.get("api_key_env", default_key_env)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise GenerationConfigError(
            f"Generation provider {provider!r} is configured but the API key env "
            f"var {key_env!r} is not set. Export it to enable /answer."
        )

    model = gen_cfg.get("model")
    if not model:
        raise GenerationConfigError(
            f"generation.model is required for provider {provider!r}."
        )

    import importlib

    module = importlib.import_module(f".{module_name}", __package__)
    cls = getattr(module, class_name)

    kwargs = dict(
        max_tokens=int(gen_cfg.get("max_tokens", 1024)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        timeout=float(gen_cfg.get("timeout", 60.0)),
    )
    base_url = gen_cfg.get("base_url")
    if base_url:
        kwargs["base_url"] = base_url

    return cls(api_key, model, **kwargs)
