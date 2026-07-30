"""Unit tests for the search() retrieval seam (rag.query) — dense pool, rerank
reorder/trim, and fetch-width. The cross-encoder is monkeypatched and the
store is a fake (see docs/ADR-multi-corpus-profiles-and-pluggable-store.md,
Axis 2 — RetrievalStore seam), so these run offline with no model download and
no chromadb."""

import pytest

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
        self.last_text = None
        self.last_hybrid = None

    def query(self, embedding, k, where=None, *, text=None, hybrid=False):
        self.last_k = k
        self.last_where = where
        self.last_text = text
        self.last_hybrid = hybrid
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


def test_search_threads_raw_text_and_hybrid_flag_to_store(monkeypatch):
    """Phase 1: search() forwards the RAW query string as text= (never the
    embedding-prefixed variant) and the hybrid flag as-is, without changing
    results (Chroma ignores both)."""
    store = FakeStore([f"doc{i}" for i in range(5)])
    # A config with a query_instruction prefix proves text= is the raw query,
    # not the prefixed embed_input.
    cfg = {"embedding_model": "x", "query_instruction": "PREFIX: "}
    recs = q.search("my raw query", n_results=3, config=cfg, model=_fake_model(),
                    store=store, rerank=False, hybrid=False)
    assert store.last_text == "my raw query"     # raw, not "PREFIX: my raw query"
    assert store.last_hybrid is False
    assert [r["document"] for r in recs] == ["doc0", "doc1", "doc2"]


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


# ── hybrid: BM25 fusion inside search() ──────────────────────────────────────────


class FakeLexical:
    """Stand-in for rag.lexical.LexicalIndex: returns fixed lexical hits and
    records the (text, k, where) it was asked for."""

    def __init__(self, docs):
        self.docs = docs                 # list of document strings
        self.calls = []

    def query(self, text, k, where=None):
        self.calls.append((text, k, where))
        return [
            {"document": d, "metadata": {"title": d, "path": d}, "distance": None}
            for d in self.docs
        ]


def test_hybrid_fuses_lexical_pool_and_boosts_dense_straggler(monkeypatch):
    # doc9 is dead last in the dense pool (rank 10) but rank 1 in lexical — RRF
    # must pull it to the top of the fused, reranked-off result.
    store = FakeStore([f"doc{i}" for i in range(10)])
    lex = FakeLexical(["doc9"])
    monkeypatch.setattr("rag.lexical.get_lexical", lambda *a, **k: lex)

    recs = q.search("k8s pods", n_results=5, config={"embedding_model": "x"},
                    model=_fake_model(), store=store, rerank=False, hybrid=True)

    assert store.last_hybrid is True                 # flag threaded to the store
    assert store.last_k == 50                        # widened to hybrid_fetch_k default
    assert lex.calls and lex.calls[0][0] == "k8s pods"   # raw query passed to lexical
    assert recs[0]["document"] == "doc9"             # fused to #1 from dense rank 10
    assert recs[0]["rank"] == 1


def test_hybrid_skipped_when_store_supports_native_fusion(monkeypatch):
    store = FakeStore([f"doc{i}" for i in range(5)])
    store.supports_hybrid = True         # native-fusion backend
    monkeypatch.setattr(
        "rag.lexical.get_lexical",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fuse client-side")),
    )
    recs = q.search("q", n_results=3, config={"embedding_model": "x"},
                    model=_fake_model(), store=store, rerank=False, hybrid=True)
    assert [r["document"] for r in recs] == ["doc0", "doc1", "doc2"]   # store order trusted
    assert store.last_hybrid is True     # store still receives the flag


def test_hybrid_false_never_touches_lexical_and_is_byte_identical(monkeypatch):
    store = FakeStore([f"doc{i}" for i in range(5)])
    monkeypatch.setattr(
        "rag.lexical.get_lexical",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("hybrid=False must not fuse")),
    )
    recs = q.search("q", n_results=5, config={"embedding_model": "x"},
                    model=_fake_model(), store=store, rerank=False, hybrid=False)
    assert [r["document"] for r in recs] == ["doc0", "doc1", "doc2", "doc3", "doc4"]
    assert store.last_hybrid is False


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

    def query(self, embedding, k, where=None, *, text=None, hybrid=False):
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
        def query(self, embedding, k, where=None, *, text=None, hybrid=False):
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


# ── rerank_default: per-profile default for callers above search() ──────────────


def test_rerank_default_reads_config_key():
    assert q.rerank_default({"rerank_default": False}) is False
    assert q.rerank_default({"rerank_default": True}) is True


def test_rerank_default_falls_back_to_true_when_absent():
    """Configs predating the key keep the old default-on behaviour."""
    assert q.rerank_default({}) is True
    assert q.rerank_default(None) is True


def _run_cli(monkeypatch, argv, config):
    """Run q.main() with search() stubbed; return the kwargs it was called with."""
    calls = {}

    def fake_search(query, n_results=8, **kw):
        calls.update(kw)
        return []

    monkeypatch.setattr(q, "load_config", lambda: config)
    monkeypatch.setattr(q, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(q, "search", fake_search)
    monkeypatch.setattr("sys.argv", argv)
    q.main()
    return calls


def test_cli_rerank_follows_config_when_no_flag(monkeypatch):
    calls = _run_cli(monkeypatch, ["rag-query", "hello"],
                     {"embedding_model": "x", "rerank_default": False})
    assert calls["rerank"] is False

    calls = _run_cli(monkeypatch, ["rag-query", "hello"],
                     {"embedding_model": "x", "rerank_default": True})
    assert calls["rerank"] is True


def test_cli_flags_override_config_default(monkeypatch):
    """--rerank / --no-rerank beat rerank_default in both directions."""
    calls = _run_cli(monkeypatch, ["rag-query", "hello", "--rerank"],
                     {"embedding_model": "x", "rerank_default": False})
    assert calls["rerank"] is True

    calls = _run_cli(monkeypatch, ["rag-query", "hello", "--no-rerank"],
                     {"embedding_model": "x", "rerank_default": True})
    assert calls["rerank"] is False


def test_cli_rerank_flags_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(q, "load_config", lambda: {"embedding_model": "x"})
    monkeypatch.setattr(q, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(q, "search", lambda *a, **k: [])
    monkeypatch.setattr("sys.argv", ["rag-query", "hi", "--rerank", "--no-rerank"])
    with pytest.raises(SystemExit):
        q.main()
