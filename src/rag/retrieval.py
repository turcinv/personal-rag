"""Backend-neutral retrieval request filter.

Application layers (CLI, API, MCP, answer generation, eval) describe *what* to
filter with this typed DTO; they never construct backend query syntax. Each
backend compiles the DTO into its own dialect:

- ``ChromaStore`` compiles it to a Chroma ``$eq``/``$and`` where-dict
  (``rag.store.compile_where``) — the only place that syntax is built.
- the lexical BM25 adapter evaluates it with the neutral ``matches_scalar``
  predicate.

Scalar constraints are exact-equality on a single metadata field. ``tags`` is
deliberately separate: it is a post-filter applied inside ``query.search`` over
the comma-joined ``tags`` metadata string (exact, case-insensitive membership;
multiple tags = AND), because Chroma 0.6.x cannot filter that string natively.

The scalar field order — domain, subdomain, type, source, confidence, status —
is significant: it is the order clauses are emitted into the compiled where-dict,
preserved from the original ``build_where`` for byte-identical filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

# Canonical scalar field order — do not reorder (defines compiled clause order).
SCALAR_FIELDS: Tuple[str, ...] = (
    "domain",
    "subdomain",
    "type",
    "source",
    "confidence",
    "status",
)


@dataclass(frozen=True)
class RetrievalFilter:
    """A backend-independent metadata filter for one retrieval request."""

    domain: Optional[str] = None
    subdomain: Optional[str] = None
    type: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[str] = None
    status: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def create(
        cls,
        *,
        domain=None,
        subdomain=None,
        type=None,
        source=None,
        confidence=None,
        status=None,
        tags=None,
    ) -> "RetrievalFilter":
        """Build a filter, normalizing falsy scalars to ``None`` and tags to a tuple.

        Falsy handling mirrors the historical ``build_where`` (``if value:``): an
        empty string is treated as "no constraint", so callers can forward
        optional request fields directly.
        """
        return cls(
            domain=domain or None,
            subdomain=subdomain or None,
            type=type or None,
            source=source or None,
            confidence=confidence or None,
            status=status or None,
            tags=tuple(tags) if tags else (),
        )

    @classmethod
    def from_api_filters(cls, filters: Any) -> "RetrievalFilter":
        """Build from an API ``QueryFilters`` model (or ``None``).

        The single consolidation point for the ``/query`` and ``/answer`` routes,
        so both map request fields identically. The API JSON field ``type`` maps
        straight to the DTO's ``type``.
        """
        if filters is None:
            return cls()
        return cls.create(
            domain=filters.domain,
            subdomain=filters.subdomain,
            type=filters.type,
            source=filters.source,
            confidence=filters.confidence,
            status=filters.status,
            tags=filters.tags,
        )

    def scalar_constraints(self) -> List[Tuple[str, str]]:
        """Return ``(field, value)`` equality pairs in canonical order."""
        pairs = []
        for name in SCALAR_FIELDS:
            value = getattr(self, name)
            if value:
                pairs.append((name, value))
        return pairs

    @property
    def has_scalar(self) -> bool:
        return bool(self.scalar_constraints())

    @property
    def is_empty(self) -> bool:
        return not self.has_scalar and not self.tags

    def matches_scalar(self, metadata: Mapping[str, Any]) -> bool:
        """Neutral predicate: every scalar constraint equals the metadata field."""
        for name, value in self.scalar_constraints():
            if (metadata or {}).get(name) != value:
                return False
        return True

    def normalized_tags(self) -> set:
        """Lower-cased, stripped, non-empty tag set for the post-filter."""
        return {tag.strip().lower() for tag in self.tags if tag and tag.strip()}
