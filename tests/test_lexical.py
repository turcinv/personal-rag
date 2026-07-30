"""Unit tests for the BM25 lexical index + RRF fusion (rag.lexical).

Offline and dependency-free: uses a tiny temp SQLite FTS5 index (stdlib) and
plain dicts — no chromadb, no embedding model, no network.
"""

import pytest

from rag.lexical import (
    LexicalIndex,
    _matches_where,
    _to_match_query,
    rrf_fuse,
)


# ── FTS5 MATCH sanitization ──────────────────────────────────────────────────


def test_to_match_query_quotes_and_or_joins_tokens():
    assert _to_match_query("Kubernetes pod") == '"kubernetes" OR "pod"'


def test_to_match_query_empty_and_punctuation_return_none():
    # The crash guard: MATCH '' raises OperationalError, so these must be None so
    # LexicalIndex.query can short-circuit to [] before touching SQL.
    for junk in ("", "   ", "!!!", "?-.,;", "🙂🙂"):
        assert _to_match_query(junk) is None


def test_to_match_query_neutralizes_fts5_keywords():
    # AND/OR/NOT/NEAR become quoted string literals, not operators — no syntax err.
    assert _to_match_query("AND OR NOT NEAR foo") == \
        '"and" OR "or" OR "not" OR "near" OR "foo"'


def test_to_match_query_keeps_unicode_token_whole():
    # Python 3 \w is Unicode-aware, so an accented word stays a single token.
    assert _to_match_query("nasazení") == '"nasazení"'


# ── LexicalIndex build + query ───────────────────────────────────────────────


def _build(tmp_path, rows):
    """rows: list of (chunk_id, document, metadata). Returns an opened index."""
    path = tmp_path / "lex.db"
    n = LexicalIndex.build(iter(rows), path)
    assert n == len(rows)
    return LexicalIndex(str(path))


def test_bm25_ranks_term_match_above_non_match(tmp_path):
    idx = _build(tmp_path, [
        ("a", "kubernetes pod pending troubleshooting", {"title": "A"}),
        ("b", "python packaging and virtualenvs", {"title": "B"}),
        ("c", "a short note about pods in kubernetes", {"title": "C"}),
    ])
    hits = idx.query("kubernetes pod", 10)
    docs = [h["metadata"]["title"] for h in hits]
    assert "B" not in docs             # no term overlap → not returned
    assert docs[0] == "A"              # densest term match ranks first
    assert set(docs) == {"A", "C"}


def test_query_empty_tokens_returns_empty_without_crashing(tmp_path):
    idx = _build(tmp_path, [("a", "hello world", {"title": "A"})])
    assert idx.query("", 10) == []
    assert idx.query("!!!", 10) == []
    assert idx.query("   ", 10) == []


def test_query_fts5_keywords_do_not_raise(tmp_path):
    idx = _build(tmp_path, [("a", "the AND gate and the OR gate", {"title": "A"})])
    # Must not raise a syntax error; whether it matches is incidental.
    hits = idx.query("AND OR NOT NEAR gate", 10)
    assert [h["metadata"]["title"] for h in hits] == ["A"]


def test_query_diacritics_folded(tmp_path):
    # unicode61 folds diacritics: the accented and unaccented forms both match.
    idx = _build(tmp_path, [("a", "nasazení aplikace do produkce", {"title": "A"})])
    assert [h["metadata"]["title"] for h in idx.query("nasazeni", 10)] == ["A"]
    assert [h["metadata"]["title"] for h in idx.query("nasazení", 10)] == ["A"]


def test_query_returns_metadata_roundtrip_and_none_distance(tmp_path):
    meta = {"title": "T", "path": "p/x.md", "domain": "DevOps", "confidence": "high"}
    idx = _build(tmp_path, [("a", "kubernetes deployment", meta)])
    hits = idx.query("kubernetes", 10)
    assert len(hits) == 1
    assert hits[0]["metadata"] == meta          # exact JSON round-trip
    assert hits[0]["distance"] is None          # lexical hit carries no cosine dist
    assert hits[0]["document"] == "kubernetes deployment"


def test_query_where_post_filter_drops_wrong_domain(tmp_path):
    idx = _build(tmp_path, [
        ("a", "kubernetes networking", {"title": "A", "domain": "DevOps"}),
        ("b", "kubernetes networking", {"title": "B", "domain": "Security"}),
    ])
    # NB: identical text can't actually coexist in a real content-hash collection;
    # here it just exercises the where post-filter deterministically.
    hits = idx.query("kubernetes networking", 10, where={"domain": {"$eq": "DevOps"}})
    assert [h["metadata"]["title"] for h in hits] == ["A"]


# ── _matches_where interpreter ───────────────────────────────────────────────


def test_matches_where_none_matches_everything():
    assert _matches_where({"domain": "X"}, None) is True
    assert _matches_where({"domain": "X"}, {}) is True


def test_matches_where_eq_and_and():
    meta = {"domain": "DevOps", "status": "processed"}
    assert _matches_where(meta, {"domain": {"$eq": "DevOps"}}) is True
    assert _matches_where(meta, {"domain": {"$eq": "Security"}}) is False
    assert _matches_where(meta, {"$and": [
        {"domain": {"$eq": "DevOps"}}, {"status": {"$eq": "processed"}},
    ]}) is True
    assert _matches_where(meta, {"$and": [
        {"domain": {"$eq": "DevOps"}}, {"status": {"$eq": "draft"}},
    ]}) is False


# ── rrf_fuse ─────────────────────────────────────────────────────────────────


def _rec(doc, dist):
    return {"document": doc, "metadata": {"title": doc}, "distance": dist}


DENSE = [_rec("a", 0.1), _rec("b", 0.2), _rec("c", 0.3)]       # dense ranks a,b,c
LEXICAL = [_rec("c", None), _rec("d", None)]                    # lexical ranks c,d


def _docs(recs):
    return [r["document"] for r in recs]


def test_rrf_default_weights_fuse_and_boost_shared_doc():
    fused = rrf_fuse(DENSE, LEXICAL, weights=(1.0, 1.0), k_rrf=60)
    # c is in both pools → highest fused score; then a (dense#1), then the b/d tie
    # is broken deterministically by dense-rank (b has one, d does not → b first).
    assert _docs(fused) == ["c", "a", "b", "d"]


def test_rrf_prefers_dense_record_for_shared_doc():
    fused = rrf_fuse(DENSE, LEXICAL, weights=(1.0, 1.0), k_rrf=60)
    c = next(r for r in fused if r["document"] == "c")
    assert c["distance"] == 0.3        # the DENSE record (real distance), not None


def test_rrf_weight_0_1_is_dense_only_order_at_top():
    # weights=(w_lexical=0, w_dense=1): dense docs keep dense order; lexical-only
    # docs get score 0 and sink to the tail.
    fused = _docs(rrf_fuse(DENSE, LEXICAL, weights=(0.0, 1.0), k_rrf=60))
    assert fused[:3] == ["a", "b", "c"]
    assert fused[-1] == "d"


def test_rrf_weight_1_0_is_lexical_only_order_at_top():
    fused = _docs(rrf_fuse(DENSE, LEXICAL, weights=(1.0, 0.0), k_rrf=60))
    assert fused[:2] == ["c", "d"]


def test_rrf_dedup_keeps_one_record_per_document():
    fused = rrf_fuse(DENSE, LEXICAL, weights=(1.0, 1.0), k_rrf=60)
    assert len(fused) == len({r["document"] for r in fused}) == 4


def test_rrf_tiebreak_is_total_and_deterministic():
    # b (dense#2, absent from lexical) and d (lexical#2, absent from dense) have
    # the SAME fused score with equal weights (1/62 each). The tiebreak must be
    # total — resolved by dense-rank (b has one, d does not) — so their relative
    # order is fixed and does not depend on Python's sort stability / input order.
    order = _docs(rrf_fuse(DENSE, LEXICAL, weights=(1.0, 1.0), k_rrf=60))
    assert order.index("b") < order.index("d")
    # Same inputs, run again → identical output (no hidden nondeterminism).
    assert order == _docs(rrf_fuse(DENSE, LEXICAL, weights=(1.0, 1.0), k_rrf=60))
