"""Unit tests for the recall@k / MRR eval harness (rag.eval).

Covers the pure matching + aggregation logic and the well-formedness of the
committed golden set. The end-to-end retrieval path is exercised separately
against a populated index via `make eval`."""

import math

from rag.eval import (
    is_hit, first_hit_rank, aggregate, load_golden, DEFAULT_GOLDEN,
    stale_trap_passed, evaluate,
)


def _rec(title="", path=""):
    return {"document": "", "metadata": {"title": title, "path": path}, "distance": 0.0}


def test_is_hit_matches_title_substring():
    assert is_hit(_rec(title="Kubernetes Pod Pending State Troubleshooting"),
                  ["Pod Pending State"])


def test_is_hit_matches_path_substring():
    assert is_hit(_rec(path="Knowledge/DevOps/Kubernetes & Container Orchestration/K3s Edge AI Cluster Architecture.md"),
                  ["K3s Edge AI Cluster Architecture"])


def test_is_hit_case_insensitive():
    assert is_hit(_rec(title="Learning Helm"), ["learning helm"])
    assert is_hit(_rec(title="FASTAPI"), ["fastapi"])


def test_is_hit_no_match():
    assert not is_hit(_rec(title="Pi-hole Port Requirements"), ["Kubernetes"])


def test_is_hit_multiple_expected_any():
    rec = _rec(title="Dataset Sizing Strategy for RAG Evaluation")
    assert is_hit(rec, ["Retrieval Evaluation Workflow", "Dataset Sizing Strategy"])


def test_first_hit_rank():
    records = [_rec(title="wrong one"), _rec(title="also wrong"),
               _rec(title="Django Static Files Configuration")]
    assert first_hit_rank(records, ["Django Static Files"]) == 3
    assert first_hit_rank(records, ["nonexistent"]) is None


def test_aggregate_math():
    rows = [
        {"hit@5": True, "hit@10": True, "hit_rank": 1},   # RR 1.0
        {"hit@5": False, "hit@10": True, "hit_rank": 7},  # RR ~0.1429
        {"hit@5": True, "hit@10": True, "hit_rank": 3},   # RR ~0.3333
        {"hit@5": False, "hit@10": False, "hit_rank": None},  # RR 0
    ]
    agg = aggregate(rows)
    assert agg["n"] == 4
    assert agg["recall@5"] == 0.5      # 2/4
    assert agg["recall@10"] == 0.75    # 3/4
    assert math.isclose(agg["mrr"], (1.0 + 1/7 + 1/3 + 0) / 4, rel_tol=1e-3)


def test_aggregate_empty():
    assert aggregate([]) == {"n": 0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}


def test_golden_set_wellformed():
    golden = load_golden(DEFAULT_GOLDEN)
    assert 30 <= len(golden) <= 60, "golden set should hold 30-50 queries"
    kinds = set()
    for item in golden:
        assert item["query"].strip(), "empty query"
        assert isinstance(item["expected"], list) and item["expected"], "expected must be a non-empty list"
        assert item["kind"] in {"vault", "resource", "stale-trap"}, f"bad kind: {item.get('kind')}"
        if item["kind"] == "stale-trap":
            assert isinstance(item.get("must_not", []), list), "must_not must be a list"
        kinds.add(item["kind"])
    assert {"vault", "resource"} <= kinds, "golden set must cover both main corpora"


# ── stale-trap scoring (spec §3) ─────────────────────────────────────────────

class _EvalFakeStore:
    """Returns the configured titles in order as search records, ignoring the
    embedding. `search()` with rerank=False preserves this order and trims."""

    def __init__(self, titles):
        self.titles = titles

    def read_index_state(self):
        return None

    def count(self):
        return len(self.titles)

    def query(self, embedding, k, where=None, *, text=None, hybrid=False):
        return [
            {"document": t, "metadata": {"title": t, "path": t}, "distance": 0.1 * i}
            for i, t in enumerate(self.titles[:k])
        ]


def _eval_fake_model():
    class M:
        def encode(self, texts, **kw):
            import numpy as np
            return np.zeros((1, 8), dtype="float32")
    return M()


def _run_eval(monkeypatch, golden, titles):
    import rag.eval as e
    store = _EvalFakeStore(titles)
    monkeypatch.setattr(e, "get_model", lambda name: _eval_fake_model())
    monkeypatch.setattr(e, "open_store", lambda *a, **k: store)
    return evaluate(golden, n=10, config={"embedding_model": "x"})


def test_stale_trap_passed_pure_logic():
    def rec(t):
        return {"document": t, "metadata": {"title": t, "path": t}, "distance": 0.0}

    # current note first, superseded note below → passed
    recs = [rec("Current SOP"), rec("Old SOP")]
    assert stale_trap_passed(recs, ["Current SOP"], ["Old SOP"]) is True
    # superseded note above the current note → failed
    recs = [rec("Old SOP"), rec("Current SOP")]
    assert stale_trap_passed(recs, ["Current SOP"], ["Old SOP"]) is False
    # no expected hit at all → failed
    recs = [rec("Unrelated")]
    assert stale_trap_passed(recs, ["Current SOP"], ["Old SOP"]) is False
    # empty must_not → passes as soon as expected is found
    recs = [rec("Current SOP")]
    assert stale_trap_passed(recs, ["Current SOP"], []) is True


def test_stale_trap_pass_case(monkeypatch):
    golden = [{
        "query": "which SOP applies",
        "expected": ["Current SOP"],
        "must_not": ["Old SOP"],
        "kind": "stale-trap",
    }]
    result = _run_eval(monkeypatch, golden, ["Current SOP", "Old SOP"])
    st = result["stale_trap"]
    assert st["total"] == 1 and st["passed"] == 1
    assert st["rows"][0]["passed"] is True
    # stale-trap rows never enter the main recall/MRR numbers
    assert result["overall"]["n"] == 0
    assert "stale-trap" not in result["by_kind"]


def test_stale_trap_inverted_ranking_fails_and_has_teeth(monkeypatch):
    """Inverted ranking (superseded note above the current one) FAILS, and the
    must_not check is what makes it fail: the expected note IS present (so a
    scorer that ignored must_not would wrongly pass)."""
    golden = [{
        "query": "which SOP applies",
        "expected": ["Current SOP"],
        "must_not": ["Old SOP"],
        "kind": "stale-trap",
    }]
    result = _run_eval(monkeypatch, golden, ["Old SOP", "Current SOP"])
    row = result["stale_trap"]["rows"][0]
    assert row["passed"] is False
    assert result["stale_trap"]["passed"] == 0
    # Teeth: without the must_not check, this would pass — the expected note is
    # found (rank 2). The must_not note outranking it (rank 1) is the only reason
    # it fails. first_hit_rank(expected) is not None proves the "expected found"
    # half alone would greenlight it.
    assert row["expected_rank"] == 2 and row["must_not_rank"] == 1
    assert row["expected_rank"] is not None  # a must_not-blind scorer would pass


def test_main_metrics_unchanged_with_stale_trap_present(monkeypatch):
    """A stale-trap row alongside a normal vault row must not perturb the main
    recall/MRR aggregation — the vault row scores exactly as it would alone."""
    golden_with_trap = [
        {"query": "vault q", "expected": ["Vault Note"], "kind": "vault"},
        {"query": "trap q", "expected": ["Current SOP"],
         "must_not": ["Old SOP"], "kind": "stale-trap"},
    ]
    golden_without = [golden_with_trap[0]]
    titles = ["Vault Note", "Current SOP", "Old SOP"]
    with_trap = _run_eval(monkeypatch, golden_with_trap, titles)
    without = _run_eval(monkeypatch, golden_without, titles)
    assert with_trap["overall"] == without["overall"]
    assert with_trap["by_kind"] == without["by_kind"]
    assert with_trap["stale_trap"]["total"] == 1
