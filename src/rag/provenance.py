"""Vector-index provenance and generation compatibility checks."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


PROVENANCE_SCHEMA_VERSION = 1
CHUNKER_VERSION = "heading-paragraph-v1"


class ProvenanceError(RuntimeError):
    """Raised when an index cannot safely be used with the active profile."""


@dataclass(frozen=True)
class IndexProvenance:
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    normalized: bool
    chunker_version: str
    chunk_max_chars: int
    chunk_overlap_chars: int
    metric: str
    corpus_profile: str
    schema_version: int = PROVENANCE_SCHEMA_VERSION

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndexProvenance":
        return cls(**dict(value))


@dataclass(frozen=True)
class IndexState:
    generation_id: str
    provenance: IndexProvenance
    updated_at: str

    @property
    def provenance_fingerprint(self) -> str:
        return self.provenance.fingerprint

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "updated_at": self.updated_at,
            "provenance_fingerprint": self.provenance_fingerprint,
            "provenance": asdict(self.provenance),
        }

    @classmethod
    def create(cls, provenance: IndexProvenance) -> "IndexState":
        return cls(
            generation_id=uuid.uuid4().hex,
            provenance=provenance,
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndexState":
        provenance = IndexProvenance.from_dict(value["provenance"])
        state = cls(
            generation_id=str(value["generation_id"]),
            provenance=provenance,
            updated_at=str(value.get("updated_at", "")),
        )
        stored_fingerprint = value.get("provenance_fingerprint")
        if stored_fingerprint and stored_fingerprint != state.provenance_fingerprint:
            raise ProvenanceError("Stored index provenance fingerprint is corrupt")
        return state


def model_dimension(model, fallback: int = 0) -> int:
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        dimension = getter()
        if dimension is not None:
            return int(dimension)
    return int(fallback)


def expected_provenance(
    config: Mapping[str, Any],
    model=None,
    collection_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> IndexProvenance:
    name = collection_name or str(
        config.get("collection_name", "obsidian_markdown")
    )
    if dimension is None:
        dimension = model_dimension(model, int(config.get("embedding_dimension", 0)))
    return IndexProvenance(
        embedding_model=str(
            config.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
        ),
        embedding_revision=str(config.get("embedding_revision", "")),
        embedding_dimension=int(dimension),
        normalized=True,
        chunker_version=CHUNKER_VERSION,
        chunk_max_chars=int(config.get("chunk_max_chars", 1200)),
        chunk_overlap_chars=int(config.get("chunk_overlap_chars", 150)),
        metric="cosine",
        corpus_profile=str(config.get("corpus_profile", name)),
    )


def read_index_state(store) -> Optional[IndexState]:
    reader = getattr(store, "read_index_state", None)
    if not callable(reader):
        return None
    raw = reader()
    return IndexState.from_dict(raw) if raw else None


def ensure_compatible(
    store,
    expected: IndexProvenance,
    *,
    adopt_missing: bool = False,
) -> Optional[IndexState]:
    """Validate persisted provenance, optionally adopting one legacy index."""
    reader = getattr(store, "read_index_state", None)
    writer = getattr(store, "write_index_state", None)
    if not callable(reader):
        return None  # lightweight test doubles and future adapters opt in explicitly

    state = read_index_state(store)
    if state is None:
        if int(store.count()) == 0:
            return None
        if adopt_missing:
            if not callable(writer):
                raise ProvenanceError("Store cannot persist adopted index provenance")
            state = IndexState.create(expected)
            writer(state.as_dict())
            return state
        raise ProvenanceError(
            "Existing index has no provenance. Rebuild into a new collection or "
            "rerun rag-index with --adopt-index-provenance only after verifying "
            "the existing vectors were built with this exact profile."
        )

    if state.provenance_fingerprint != expected.fingerprint:
        differences = []
        actual_values = asdict(state.provenance)
        expected_values = asdict(expected)
        for field, expected_value in expected_values.items():
            actual_value = actual_values.get(field)
            if actual_value != expected_value:
                differences.append(
                    f"{field}: index={actual_value!r}, config={expected_value!r}"
                )
        raise ProvenanceError(
            "Index provenance mismatch; use a new collection or explicit rebuild. "
            + "; ".join(differences)
        )
    return state


def begin_generation(store, provenance: IndexProvenance) -> IndexState:
    """Invalidate generation-bound sidecars before the first index mutation."""
    state = IndexState.create(provenance)
    writer = getattr(store, "write_index_state", None)
    if callable(writer):
        writer(state.as_dict())
    return state


def publish_generation(
    store,
    provenance: IndexProvenance,
    previous: Optional[IndexState],
    changed: bool,
) -> IndexState:
    """Persist a new generation only when indexed content/metadata changed."""
    state = previous if previous is not None and not changed else IndexState.create(provenance)
    writer = getattr(store, "write_index_state", None)
    if not callable(writer):
        return state
    current = read_index_state(store)
    if current != state:
        writer(state.as_dict())
    return state
