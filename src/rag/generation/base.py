"""``Generator`` — the pluggable answer-synthesis protocol.

The generation layer sits ABOVE retrieval: ``rag.query.search`` returns the
relevant chunks, and a ``Generator`` turns (question + those chunks) into a
grounded, cited natural-language answer. It is orthogonal to the
``RetrievalStore`` backend (see
``docs/ADR-multi-corpus-profiles-and-pluggable-store.md``) — the same generator
works whether retrieval came from Chroma or a future OpenSearch store, because
it only ever sees ``search()``'s record dicts, never the store.

Every backend (``AnthropicGenerator``, ``OpenAIGenerator``) implements this
Protocol and reads its API key from the environment at construction time, so
the API key never lives in config files or in the request. Prompt construction
(``build_prompt`` / ``format_contexts``) is shared here so both providers ground
and cite identically.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


class GenerationError(RuntimeError):
    """Raised when a provider call fails (transport, HTTP status, or bad body).

    The API layer maps this to a 502 so a downstream LLM outage is reported as
    an upstream-dependency failure, never a 500 that looks like our bug.
    """


class GenerationConfigError(ValueError):
    """Raised at construction when generation is misconfigured.

    Two distinct causes: an unknown ``provider`` (a typo — a hard error), or a
    missing API-key env var (an expected "not wired up here" state). The API
    lifespan catches the missing-key case and serves ``/query`` normally while
    ``/answer`` returns 503; an unknown provider is a real config bug and
    surfaces loudly.
    """


# Grounding contract shared by both providers. Kept deliberately strict: answer
# only from the numbered context, cite with [n], and admit when the context does
# not contain the answer rather than inventing one.
DEFAULT_SYSTEM_PROMPT = (
    "You are a precise assistant answering questions from a personal technical "
    "knowledge base. Answer ONLY from the numbered context passages provided. "
    "Cite every claim with its passage number in square brackets, e.g. [1] or "
    "[2][3]. If the context does not contain the answer, say so plainly instead "
    "of guessing. Be concise and technical; do not pad the answer."
)


@dataclass
class AnswerResult:
    """A generated answer plus provenance for the response envelope.

    ``answer`` is the model's text. ``model`` echoes the model that actually
    served the request (providers may resolve an alias). ``usage`` is the raw
    provider token-usage dict when available (``None`` otherwise), and
    ``stop_reason`` is the provider's stop/finish reason for debugging.
    """

    answer: str
    model: str
    usage: Optional[dict] = None
    stop_reason: Optional[str] = None
    raw: dict = field(default_factory=dict, repr=False)


@runtime_checkable
class Generator(Protocol):
    """Uniform interface over one configured LLM backend.

    A generator instance is bound to one provider + model (resolved at
    construction via ``get_generator()``). It is stateless per call and safe to
    hold resident in the API's ``app.state`` for the process lifetime.
    """

    @property
    def model(self) -> str:
        """The model id this generator sends to the provider."""
        ...

    @property
    def provider(self) -> str:
        """Backend name (``"anthropic"`` / ``"openai"``) for the response envelope."""
        ...

    def generate(
        self,
        question: str,
        contexts: list,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AnswerResult:
        """Synthesize a grounded answer to ``question`` from ``contexts``.

        ``contexts`` is the list of record dicts from ``rag.query.search`` (each
        has ``document`` + ``metadata``); the shared ``build_prompt`` renders
        them into a numbered, citable block. ``max_tokens`` / ``temperature``
        override the configured defaults for this one call. Raises
        :class:`GenerationError` on any provider failure.
        """
        ...


def format_contexts(contexts: list) -> str:
    """Render ``search()`` records into a numbered, citable context block.

    Each passage becomes ``[n] Title — path`` followed by the full chunk text.
    The number ``n`` is 1-based and MUST line up with the citation indices the
    response builder emits, so the model's ``[n]`` markers resolve to the right
    source. Uses the full ``document`` text (never a truncated display copy).
    """
    blocks = []
    for i, rec in enumerate(contexts, start=1):
        meta = rec.get("metadata") or {}
        title = meta.get("title") or meta.get("path") or "untitled"
        path = meta.get("path") or ""
        header = f"[{i}] {title}"
        if path and path != title:
            header += f" — {path}"
        blocks.append(f"{header}\n{rec.get('document', '')}".strip())
    return "\n\n".join(blocks)


def build_prompt(question: str, contexts: list) -> str:
    """Build the user-turn prompt: numbered context block + the question.

    The system-turn grounding rules live in :data:`DEFAULT_SYSTEM_PROMPT`; this
    is only the user content. Both providers send the same two turns so their
    answers ground and cite identically.
    """
    context_block = format_contexts(contexts)
    return (
        f"Context passages:\n\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above, citing each claim with its [n]."
    )
