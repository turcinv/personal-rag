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
