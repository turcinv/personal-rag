"""Unit tests for the search() retrieval seam (rag.query) — dense pool, rerank
reorder/trim, and fetch-width. The cross-encoder is monkeypatched and the
store is a fake (see docs/ADR-multi-corpus-profiles-and-pluggable-store.md,
Axis 2 — RetrievalStore seam), so these run offline with no model download and
no chromadb."""

import rag.query as q


class FakeStore:
    """RetrievalStore stand-in: returns `k` deterministic records and records
    the `k` (and `where`) it was asked for. Mirrors ChromaStore.query()'s
    return shape directly — list[{"document","metadata","distance"}] — with
    no chromadb dict-of-lists unwrapping involved."""

    def __init__(self, docs):
        self.docs = docs
        self.last_k = None
        self.last_where = None

    def query(self, embedding, k, where=None):
        self.last_k = k
        self.last_where = where
        docs = self.docs[:k]
        return [
            {"document": d, "metadata": {"title": d, "path": d}, "distance": 0.1 * i}
            for i, d in enumerate(docs)
        ]


class FakeReranker:
    """Puts the doc containing 'match' first; all others score below it."""
    def predict(self, pairs):
        return [100.0 if "match" in doc else -float(i) for i, (_, doc) in enumerate(pairs)]


CFG = {"embedding_model": "x", "rerank_fetch_k": 20}


def _fake_model():
    class M:
        def encode(self, texts, **kw):
            import numpy as np
            return np.zeros((1, 8), dtype="float32")
    return M()


def test_dense_only_preserves_order_and_trims(monkeypatch):
    store = FakeStore([f"doc{i}" for i in range(20)])
    recs = q.search("q", n_results=5, config=CFG, model=_fake_model(),
                    store=store, rerank=False)
    assert [r["document"] for r in recs] == ["doc0", "doc1", "doc2", "doc3", "doc4"]
    assert store.last_k == 5             # no widening when rerank off
    assert all("rerank_score" not in r for r in recs)
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]


def test_rerank_widens_pool_reorders_and_trims(monkeypatch):
    monkeypatch.setattr(q, "get_reranker", lambda name: FakeReranker())
    docs = [f"doc{i}" for i in range(19)] + ["the match doc"]  # relevant one last
    store = FakeStore(docs)
    recs = q.search("q", n_results=5, config=CFG, model=_fake_model(),
                    store=store, rerank=True)
    assert store.last_k == 20            # widened to rerank_fetch_k
    assert len(recs) == 5                # trimmed to n_results
    assert recs[0]["document"] == "the match doc"   # reranker pulled it to #1
    assert recs[0]["rank"] == 1 and "rerank_score" in recs[0]


def test_rerank_fetch_k_at_least_n_results(monkeypatch):
    monkeypatch.setattr(q, "get_reranker", lambda name: FakeReranker())
    store = FakeStore([f"doc{i}" for i in range(50)])
    q.search("q", n_results=30, config={"embedding_model": "x", "rerank_fetch_k": 20},
             model=_fake_model(), store=store, rerank=True)
    assert store.last_k == 30            # max(n_results, rerank_fetch_k)


# ── build_where: status (native $eq) ─────────────────────────────────────────────


def test_build_where_status_emits_eq():
    assert q.build_where(status="processed") == {"status": {"$eq": "processed"}}


def test_build_where_status_and_domain_combine_with_and():
    assert q.build_where(domain="DevOps", status="processed") == {
        "$and": [{"domain": {"$eq": "DevOps"}}, {"status": {"$eq": "processed"}}]
    }


def test_build_where_all_none_returns_none():
    # Regression-critical: no filter supplied must stay None (unchanged ranking).
    assert q.build_where() is None


# ── search: tags post-filter + fetch widening ────────────────────────────────────


class FakeStoreWithTags:
    """Like FakeStore but each doc carries a comma-joined `tags` metadata string."""

    def __init__(self, docs_tags):
        self.docs_tags = docs_tags       # list of (doc, tags_string)
        self.last_k = None

    def query(self, embedding, k, where=None):
        self.last_k = k
        picked = self.docs_tags[:k]
        return [
            {"document": d, "metadata": {"title": d, "path": d, "tags": t}, "distance": 0.1 * i}
            for i, (d, t) in enumerate(picked)
        ]


def test_tags_keep_only_exact_superset_case_insensitive():
    store = FakeStoreWithTags([
        ("a", "devops, ci-cd"),   # has devops
        ("b", "python"),          # no devops -> drop
        ("c", "DevOps, k8s"),     # case-insensitive match
        ("d", ""),                # empty tags -> drop, never raise
    ])
    recs = q.search("q", n_results=10, tags=["devops"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    store=store, rerank=False)
    assert [r["document"] for r in recs] == ["a", "c"]


def test_tag_match_is_exact_not_substring():
    # `--tag ci` must NOT match a chunk tagged "ci-cd".
    store = FakeStoreWithTags([("a", "ci-cd"), ("b", "ci")])
    recs = q.search("q", n_results=10, tags=["ci"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    store=store, rerank=False)
    assert [r["document"] for r in recs] == ["b"]


def test_multiple_tags_are_and():
    store = FakeStoreWithTags([
        ("a", "devops, kubernetes"),  # both -> keep
        ("b", "devops"),              # only one -> drop
        ("c", "kubernetes"),          # only one -> drop
    ])
    recs = q.search("q", n_results=10, tags=["devops", "kubernetes"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    store=store, rerank=False)
    assert [r["document"] for r in recs] == ["a"]


def test_tags_missing_metadata_never_raises_and_drops():
    # A record whose metadata has no `tags` key at all must be dropped, not crash.
    class NoTagsStore(FakeStore):
        def query(self, embedding, k, where=None):
            self.last_k = k
            docs = self.docs[:k]
            return [
                {"document": d, "metadata": {"title": d, "path": d}, "distance": 0.1 * i}
                for i, d in enumerate(docs)  # no "tags" key
            ]

    store = NoTagsStore(["a", "b"])
    recs = q.search("q", n_results=10, tags=["devops"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    store=store, rerank=False)
    assert recs == []


def test_tags_widen_fetch_pool_to_tag_fetch_k():
    store = FakeStoreWithTags([(f"doc{i}", "devops") for i in range(5)])
    q.search("q", n_results=5, tags=["devops"],
             config={"embedding_model": "x", "tag_fetch_k": 200},
             model=_fake_model(), store=store, rerank=False)
    assert store.last_k == 200           # widened because tags supplied


def test_no_tags_does_not_widen_even_with_tag_fetch_k_configured():
    # Regression-critical: tag_fetch_k present but tags=None must not widen.
    store = FakeStore([f"doc{i}" for i in range(20)])
    q.search("q", n_results=5,
             config={"embedding_model": "x", "tag_fetch_k": 200},
             model=_fake_model(), store=store, rerank=False)
    assert store.last_k == 5


# ── CLI wiring: --status / repeated --tag ─────────────────────────────────────────


def test_cli_status_and_repeated_tag_wire_through(monkeypatch, capsys):
    """argparse accepts --status + repeated --tag; main() forwards status via the
    real build_where and the tag list straight into search()."""
    calls = {}

    def fake_search(query, n_results=8, *, filters=None, tags=None, **kw):
        calls["filters"] = filters
        calls["tags"] = tags
        return []

    monkeypatch.setattr(q, "load_config", lambda: {"embedding_model": "x"})
    monkeypatch.setattr(q, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(q, "search", fake_search)
    monkeypatch.setattr(
        "sys.argv",
        ["rag-query", "hello", "--status", "processed", "--tag", "devops", "--tag", "ci"],
    )
    q.main()
    assert calls["tags"] == ["devops", "ci"]
    assert calls["filters"] == {"status": {"$eq": "processed"}}
