"""Recall@k / MRR evaluation harness for the RAG retrieval layer.

Runs a labeled golden set (``tests/eval/golden_queries.jsonl``) through the same
:func:`rag.query.search` path the CLI uses, then reports recall@5, recall@10 and
MRR overall and per corpus (vault vs resource). A query "hits" when any of its
``expected`` substrings appears (case-insensitive) in a returned chunk's
``title`` or ``path`` metadata — robust to chunk granularity.

Use it to measure every retrieval change against a fixed baseline; see the
roadmap in CLAUDE.md. Runnable via ``make eval`` / ``scripts/eval_recall.py``.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .utils import load_config, setup_logging
from .query import build_where, get_model, open_store, search

logger = logging.getLogger("rag")

DEFAULT_GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "eval" / "golden_queries.jsonl"


def _norm(s):
    return (s or "").lower()


def load_golden(path):
    """Read the golden-query JSONL into a list of dicts."""
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(json.loads(line))
    return items


def is_hit(record, expected):
    """True if any expected substring matches the record's title or path."""
    meta = record.get("metadata", {})
    hay = _norm(meta.get("title", "")) + "\n" + _norm(meta.get("path", ""))
    return any(_norm(e) in hay for e in expected)


def first_hit_rank(records, expected):
    """1-based rank of the first hit, or None if no record matches."""
    for i, rec in enumerate(records, start=1):
        if is_hit(rec, expected):
            return i
    return None


def aggregate(rows):
    """Aggregate per-query rows into recall@5, recall@10 and MRR.

    Each row is a dict with ``hit@5`` / ``hit@10`` (bool) and ``hit_rank``
    (1-based int or None). MRR uses the reciprocal of the first-hit rank (0 for
    a miss)."""
    m = len(rows)
    if m == 0:
        return {"n": 0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0}
    r5 = sum(r["hit@5"] for r in rows) / m
    r10 = sum(r["hit@10"] for r in rows) / m
    mrr = sum((1.0 / r["hit_rank"]) if r["hit_rank"] else 0.0 for r in rows) / m
    return {"n": m, "recall@5": round(r5, 4), "recall@10": round(r10, 4), "mrr": round(mrr, 4)}


def evaluate(golden, *, n=10, config=None, collection_name=None,
             rerank=False, hybrid=False):
    """Run every golden query and return an aggregate + per-query result dict."""
    if config is None:
        config = load_config()
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    model = get_model(model_name)
    store = open_store(config, collection_name)
    count = store.count()
    if count == 0:
        logger.warning("Collection %r is EMPTY — recall will be 0. Build an index first.",
                       collection_name or config.get("collection_name"))

    per_query = []
    for item in golden:
        q, expected, kind = item["query"], item["expected"], item.get("kind", "")
        filters = None  # golden queries are unfiltered by design
        records = search(
            q, n, filters=filters, config=config, model=model,
            store=store, rerank=rerank, hybrid=hybrid,
        )
        rank = first_hit_rank(records, expected)
        per_query.append({
            "query": q,
            "kind": kind,
            "expected": expected,
            "hit_rank": rank,
            "hit@5": bool(rank and rank <= 5),
            "hit@10": bool(rank and rank <= 10),
        })

    by_kind = {}
    for kind in sorted({r["kind"] for r in per_query}):
        by_kind[kind] = aggregate([r for r in per_query if r["kind"] == kind])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "collection": collection_name or config.get("collection_name", "obsidian_markdown"),
        "collection_count": count,
        "retrieval_depth": n,
        "rerank": rerank,
        "hybrid": hybrid,
        "overall": aggregate(per_query),
        "by_kind": by_kind,
        "per_query": per_query,
    }


def print_report(result, label=None):
    """Pretty-print an evaluate() result to stdout."""
    o = result["overall"]
    print()
    if label:
        print(f"Label:      {label}")
    print(f"Model:      {result['model']}")
    print(f"Collection: {result['collection']} ({result['collection_count']} chunks)")
    print(f"Depth:      top-{result['retrieval_depth']}  |  rerank={result['rerank']}  hybrid={result['hybrid']}")
    print("=" * 78)
    print(f"{'query':<58}{'kind':<10}{'rank':>6}")
    print("-" * 78)
    for r in result["per_query"]:
        rank = r["hit_rank"] if r["hit_rank"] else "miss"
        q = r["query"] if len(r["query"]) <= 56 else r["query"][:55] + "…"
        print(f"{q:<58}{r['kind']:<10}{str(rank):>6}")
    print("-" * 78)
    print(f"OVERALL  (n={o['n']})   recall@5={o['recall@5']:.3f}  "
          f"recall@10={o['recall@10']:.3f}  MRR={o['mrr']:.3f}")
    for kind, a in result["by_kind"].items():
        print(f"  {kind:<8} (n={a['n']})   recall@5={a['recall@5']:.3f}  "
              f"recall@10={a['recall@10']:.3f}  MRR={a['mrr']:.3f}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Recall@k / MRR eval for RAG retrieval.")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN),
                        help="Path to golden_queries.jsonl")
    parser.add_argument("-n", "--n-results", type=int, default=10, dest="n",
                        help="Retrieval depth (must be >= 10 for recall@10; default 10)")
    parser.add_argument("--collection", default=None,
                        help="Override the collection name from config.yaml")
    parser.add_argument("--label", default=None,
                        help="Free-text label recorded in the output (e.g. 'baseline')")
    parser.add_argument("--out", default=None,
                        help="Write the full result JSON to this path")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Disable cross-encoder reranking (dense retrieval only)")
    parser.add_argument("--hybrid", dest="hybrid", action="store_true",
                        help="Enable BM25+dense hybrid fusion (Phase 3d)")
    parser.set_defaults(rerank=True, hybrid=False)
    args = parser.parse_args()

    config = load_config()
    setup_logging(config, console=False)

    if args.n < 10:
        logger.warning("retrieval depth %d < 10; recall@10 will be capped", args.n)

    golden = load_golden(args.golden)
    result = evaluate(golden, n=args.n, config=config, collection_name=args.collection,
                      rerank=args.rerank, hybrid=args.hybrid)
    if args.label:
        result["label"] = args.label
    print_report(result, label=args.label)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
