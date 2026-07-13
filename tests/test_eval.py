"""Unit tests for the recall@k / MRR eval harness (rag.eval).

Covers the pure matching + aggregation logic and the well-formedness of the
committed golden set. The end-to-end retrieval path is exercised separately
against a populated index via `make eval`."""

import math

from rag.eval import is_hit, first_hit_rank, aggregate, load_golden, DEFAULT_GOLDEN


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
        assert item["kind"] in {"vault", "resource"}, f"bad kind: {item.get('kind')}"
        kinds.add(item["kind"])
    assert kinds == {"vault", "resource"}, "golden set must cover both corpora"
