"""Unit tests for the search() retrieval seam (rag.query) — dense pool, rerank
reorder/trim, and fetch-width. The cross-encoder is monkeypatched so these run
offline with no model download."""

import rag.query as q


class FakeColl:
    """Returns `n` deterministic docs and records the n_results it was asked for."""
    def __init__(self, docs):
        self.docs = docs
        self.last_n = None

    def query(self, **kw):
        self.last_n = kw["n_results"]
        docs = self.docs[: kw["n_results"]]
        return {
            "documents": [docs],
            "metadatas": [[{"title": d, "path": d} for d in docs]],
            "distances": [[0.1 * i for i in range(len(docs))]],
        }


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
    coll = FakeColl([f"doc{i}" for i in range(20)])
    recs = q.search("q", n_results=5, config=CFG, model=_fake_model(),
                    collection=coll, rerank=False)
    assert [r["document"] for r in recs] == ["doc0", "doc1", "doc2", "doc3", "doc4"]
    assert coll.last_n == 5              # no widening when rerank off
    assert all("rerank_score" not in r for r in recs)
    assert [r["rank"] for r in recs] == [1, 2, 3, 4, 5]


def test_rerank_widens_pool_reorders_and_trims(monkeypatch):
    monkeypatch.setattr(q, "get_reranker", lambda name: FakeReranker())
    docs = [f"doc{i}" for i in range(19)] + ["the match doc"]  # relevant one last
    coll = FakeColl(docs)
    recs = q.search("q", n_results=5, config=CFG, model=_fake_model(),
                    collection=coll, rerank=True)
    assert coll.last_n == 20             # widened to rerank_fetch_k
    assert len(recs) == 5                # trimmed to n_results
    assert recs[0]["document"] == "the match doc"   # reranker pulled it to #1
    assert recs[0]["rank"] == 1 and "rerank_score" in recs[0]


def test_rerank_fetch_k_at_least_n_results(monkeypatch):
    monkeypatch.setattr(q, "get_reranker", lambda name: FakeReranker())
    coll = FakeColl([f"doc{i}" for i in range(50)])
    q.search("q", n_results=30, config={"embedding_model": "x", "rerank_fetch_k": 20},
             model=_fake_model(), collection=coll, rerank=True)
    assert coll.last_n == 30             # max(n_results, rerank_fetch_k)


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


class FakeCollWithTags:
    """Like FakeColl but each doc carries a comma-joined `tags` metadata string."""

    def __init__(self, docs_tags):
        self.docs_tags = docs_tags       # list of (doc, tags_string)
        self.last_n = None

    def query(self, **kw):
        self.last_n = kw["n_results"]
        picked = self.docs_tags[: kw["n_results"]]
        docs = [d for d, _ in picked]
        metas = [{"title": d, "path": d, "tags": t} for d, t in picked]
        return {
            "documents": [docs],
            "metadatas": [metas],
            "distances": [[0.1 * i for i in range(len(docs))]],
        }


def test_tags_keep_only_exact_superset_case_insensitive():
    coll = FakeCollWithTags([
        ("a", "devops, ci-cd"),   # has devops
        ("b", "python"),          # no devops -> drop
        ("c", "DevOps, k8s"),     # case-insensitive match
        ("d", ""),                # empty tags -> drop, never raise
    ])
    recs = q.search("q", n_results=10, tags=["devops"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    collection=coll, rerank=False)
    assert [r["document"] for r in recs] == ["a", "c"]


def test_tag_match_is_exact_not_substring():
    # `--tag ci` must NOT match a chunk tagged "ci-cd".
    coll = FakeCollWithTags([("a", "ci-cd"), ("b", "ci")])
    recs = q.search("q", n_results=10, tags=["ci"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    collection=coll, rerank=False)
    assert [r["document"] for r in recs] == ["b"]


def test_multiple_tags_are_and():
    coll = FakeCollWithTags([
        ("a", "devops, kubernetes"),  # both -> keep
        ("b", "devops"),              # only one -> drop
        ("c", "kubernetes"),          # only one -> drop
    ])
    recs = q.search("q", n_results=10, tags=["devops", "kubernetes"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    collection=coll, rerank=False)
    assert [r["document"] for r in recs] == ["a"]


def test_tags_missing_metadata_never_raises_and_drops():
    # A record whose metadata has no `tags` key at all must be dropped, not crash.
    class NoTagsColl(FakeColl):
        def query(self, **kw):
            self.last_n = kw["n_results"]
            docs = self.docs[: kw["n_results"]]
            return {
                "documents": [docs],
                "metadatas": [[{"title": d, "path": d} for d in docs]],  # no "tags"
                "distances": [[0.1 * i for i in range(len(docs))]],
            }

    coll = NoTagsColl(["a", "b"])
    recs = q.search("q", n_results=10, tags=["devops"],
                    config={"embedding_model": "x"}, model=_fake_model(),
                    collection=coll, rerank=False)
    assert recs == []


def test_tags_widen_fetch_pool_to_tag_fetch_k():
    coll = FakeCollWithTags([(f"doc{i}", "devops") for i in range(5)])
    q.search("q", n_results=5, tags=["devops"],
             config={"embedding_model": "x", "tag_fetch_k": 200},
             model=_fake_model(), collection=coll, rerank=False)
    assert coll.last_n == 200            # widened because tags supplied


def test_no_tags_does_not_widen_even_with_tag_fetch_k_configured():
    # Regression-critical: tag_fetch_k present but tags=None must not widen.
    coll = FakeColl([f"doc{i}" for i in range(20)])
    q.search("q", n_results=5,
             config={"embedding_model": "x", "tag_fetch_k": 200},
             model=_fake_model(), collection=coll, rerank=False)
    assert coll.last_n == 5


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
