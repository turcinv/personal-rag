"""Index provenance compatibility and generation tests."""

from dataclasses import replace

import pytest

from rag.provenance import (
    IndexProvenance,
    IndexState,
    ProvenanceError,
    ensure_compatible,
    publish_generation,
)


def _provenance():
    return IndexProvenance(
        embedding_model="model-a",
        embedding_revision="rev-1",
        embedding_dimension=384,
        normalized=True,
        chunker_version="heading-paragraph-v1",
        chunk_max_chars=1200,
        chunk_overlap_chars=150,
        metric="cosine",
        corpus_profile="personal",
    )


class StateStore:
    def __init__(self, state=None, count=1):
        self.state = state.as_dict() if isinstance(state, IndexState) else state
        self._count = count
        self.writes = []

    def count(self):
        return self._count

    def read_index_state(self):
        return self.state

    def write_index_state(self, state):
        self.state = state
        self.writes.append(state)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("embedding_model", "model-b"),
        ("embedding_revision", "rev-2"),
        ("embedding_dimension", 768),
        ("normalized", False),
        ("chunker_version", "heading-paragraph-v2"),
        ("chunk_max_chars", 1800),
        ("chunk_overlap_chars", 200),
        ("metric", "l2"),
        ("corpus_profile", "other"),
        ("schema_version", 2),
    ],
)
def test_every_provenance_mismatch_is_rejected(field, value):
    actual = _provenance()
    store = StateStore(IndexState.create(actual))

    with pytest.raises(ProvenanceError, match=field):
        ensure_compatible(store, replace(actual, **{field: value}))


def test_matching_provenance_allows_metadata_only_generation_reuse():
    provenance = _provenance()
    state = IndexState.create(provenance)
    store = StateStore(state)

    compatible = ensure_compatible(store, provenance)
    published = publish_generation(store, provenance, compatible, changed=False)

    assert published == state
    assert store.writes == []


def test_changed_index_gets_new_generation_with_same_provenance():
    provenance = _provenance()
    state = IndexState.create(provenance)
    store = StateStore(state)

    published = publish_generation(store, provenance, state, changed=True)

    assert published.generation_id != state.generation_id
    assert published.provenance_fingerprint == state.provenance_fingerprint
    assert len(store.writes) == 1


def test_legacy_nonempty_index_requires_explicit_adoption():
    store = StateStore(None, count=10)

    with pytest.raises(ProvenanceError, match="no provenance"):
        ensure_compatible(store, _provenance())

    adopted = ensure_compatible(store, _provenance(), adopt_missing=True)
    assert adopted is not None
    assert len(store.writes) == 1


def test_empty_index_can_initialize_without_adoption():
    store = StateStore(None, count=0)
    assert ensure_compatible(store, _provenance()) is None
