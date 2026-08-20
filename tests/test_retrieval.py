"""Contract tests for the backend-neutral retrieval filter DTO.

Covers: DTO construction/normalization, Chroma where-compilation (byte-identical
to the legacy build_where), lexical neutral matching, and CLI/API request parity
(the same filter through both surfaces compiles to the same backend where-dict).
All offline — no model, no chromadb, no network.
"""

import rag.query as q
from rag.lexical import _where_matches
from rag.retrieval import RetrievalFilter
from rag.store import compile_where


# ── DTO construction / normalization ─────────────────────────────────────────


def test_create_normalizes_falsy_scalars_and_tags():
    rf = RetrievalFilter.create(domain="", status="processed", tags=[])
    assert rf.domain is None
    assert rf.status == "processed"
    assert rf.tags == ()
    assert rf.has_scalar
    assert not rf.is_empty


def test_empty_filter_is_empty_and_compiles_to_none():
    rf = RetrievalFilter()
    assert rf.is_empty
    assert compile_where(rf) is None


def test_scalar_constraint_order_is_canonical():
    rf = RetrievalFilter.create(
        status="s", domain="d", type="t", source="src", confidence="c", subdomain="sub"
    )
    assert rf.scalar_constraints() == [
        ("domain", "d"),
        ("subdomain", "sub"),
        ("type", "t"),
        ("source", "src"),
        ("confidence", "c"),
        ("status", "s"),
    ]


# ── Chroma compilation parity with legacy build_where ────────────────────────


def test_compile_single_clause_is_unwrapped():
    rf = RetrievalFilter.create(status="processed")
    assert compile_where(rf) == {"status": {"$eq": "processed"}}


def test_compile_multi_clause_is_and_in_field_order():
    rf = RetrievalFilter.create(domain="DevOps", status="processed")
    assert compile_where(rf) == {
        "$and": [{"domain": {"$eq": "DevOps"}}, {"status": {"$eq": "processed"}}]
    }


def test_compile_passthrough_legacy_dict():
    legacy = {"domain": {"$eq": "X"}}
    assert compile_where(legacy) is legacy
    assert compile_where(None) is None


def test_build_where_matches_compile_where():
    # The legacy shim and the DTO path must produce identical dicts.
    assert q.build_where(domain="DevOps", status="processed") == compile_where(
        RetrievalFilter.create(domain="DevOps", status="processed")
    )
    assert q.build_where() is None
    assert q.build_where(status="processed") == {"status": {"$eq": "processed"}}


def test_tags_never_appear_in_compiled_where():
    rf = RetrievalFilter.create(tags=["devops", "ci"])
    assert compile_where(rf) is None
    assert list(rf.tags) == ["devops", "ci"]


# ── Lexical neutral matching ─────────────────────────────────────────────────


def test_lexical_matches_dto_scalar_constraints():
    rf = RetrievalFilter.create(domain="DevOps")
    assert _where_matches(rf, {"domain": "DevOps"}) is True
    assert _where_matches(rf, {"domain": "Security"}) is False
    assert _where_matches(None, {"domain": "anything"}) is True


def test_lexical_matches_legacy_dict():
    where = {"$and": [{"domain": {"$eq": "DevOps"}}, {"status": {"$eq": "processed"}}]}
    assert _where_matches(where, {"domain": "DevOps", "status": "processed"}) is True
    assert _where_matches(where, {"domain": "DevOps", "status": "draft"}) is False


def test_dto_matches_scalar_ignores_tags():
    rf = RetrievalFilter.create(domain="DevOps", tags=["devops"])
    # tags are a post-filter, not part of scalar matching
    assert rf.matches_scalar({"domain": "DevOps"}) is True


# ── CLI / API request parity ─────────────────────────────────────────────────


class _ApiFilters:
    """Stand-in for the API QueryFilters pydantic model."""

    def __init__(self, **kw):
        for name in ("domain", "subdomain", "type", "source", "confidence", "status", "tags"):
            setattr(self, name, kw.get(name))


def test_cli_and_api_filters_compile_identically():
    api_rf = RetrievalFilter.from_api_filters(
        _ApiFilters(domain="DevOps", type="book", status="processed", tags=["devops"])
    )
    cli_rf = RetrievalFilter.create(
        domain="DevOps", type="book", status="processed", tags=["devops"]
    )
    assert compile_where(api_rf) == compile_where(cli_rf)
    assert list(api_rf.tags) == list(cli_rf.tags) == ["devops"]


def test_from_api_filters_none_is_empty():
    assert RetrievalFilter.from_api_filters(None).is_empty
