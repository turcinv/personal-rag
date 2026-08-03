"""
Retrieval smoke tests for personal-rag.

Run after every reindex to verify query quality across all vault domains.

Usage:
    make test                   # run all queries
    make test K=kubernetes      # filter by keyword
    .venv/bin/python tests/test_queries.py [keyword]

Each query prints the top 3 results. Distance < 0.6 is a strong match,
0.6-0.8 is acceptable, > 0.8 is weak.
"""

import sys

from rag.utils import load_config  # sets telemetry env var and patches posthog before chromadb loads

from rag.store import get_store
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Test queries — one or two per domain, plus cross-domain
# Add your own as your vault grows.
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # DevOps
    ("DevOps",              "What do I know about Kubernetes and container orchestration?"),
    ("DevOps",              "When should I use K3s instead of full Kubernetes?"),
    ("DevOps",              "How do I optimise cloud costs when learning DevOps?"),

    # AI & Automation
    ("AI & Automation",     "What tools and SDKs do I use for AI automation?"),
    ("AI & Automation",     "What have I learned about Gemini SDK and RAG?"),

    # Software Engineering
    ("Software Engineering","How should I handle secrets in Python services?"),
    ("Software Engineering","What are my notes on SQLite schema and migrations?"),
    ("Software Engineering","How do I design SMS notification services?"),

    # Bioprocessing
    ("Bioprocessing",       "What do I know about bioprocessing workflows?"),

    # Business Development
    ("Business Development","What are my notes on business development strategy?"),

    # Career Development
    ("Career Development",  "What skills should I focus on for career growth?"),

    # Cross-domain
    ("Cross-domain",        "What decisions have I documented and why?"),
    ("Cross-domain",        "What frameworks do I use for problem solving?"),
]

N_RESULTS    = 3
DISTANCE_WARN = 0.75


def run_tests(filter_keyword: str = ""):
    config = load_config()
    model = SentenceTransformer(config["embedding_model"])
    store = get_store(config)

    queries = TEST_QUERIES
    if filter_keyword:
        queries = [
            (d, q) for d, q in queries
            if filter_keyword.lower() in q.lower() or filter_keyword.lower() in d.lower()
        ]
        if not queries:
            print(f"No queries match '{filter_keyword}'")
            return

    passed = warned = 0

    for domain, query in queries:
        embedding = model.encode([query], normalize_embeddings=True).tolist()[0]
        records   = store.query(embedding, N_RESULTS)
        docs      = [r["document"] for r in records]
        metas     = [r["metadata"] for r in records]
        distances = [r["distance"] for r in records]
        best      = distances[0] if distances else 1.0

        status = "OK" if best < DISTANCE_WARN else "WEAK"
        warned += status == "WEAK"
        passed += status == "OK"

        print()
        print(f"[{domain}] [{status}]  {query}")
        print(f"  Best distance: {best:.4f}")
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            title   = meta.get("title", "?")
            heading = meta.get("heading", "")
            path    = meta.get("path", "")
            heading_str = f" — {heading}" if heading and heading != title else ""
            print(f"  {i}. [{dist:.4f}] {title}{heading_str}")
            print(f"       {path}")

    total = passed + warned
    print()
    print("=" * 60)
    print(f"Results: {passed}/{total} OK  |  {warned}/{total} WEAK (distance >= {DISTANCE_WARN})")
    print("=" * 60)


if __name__ == "__main__":
    keyword = sys.argv[1] if len(sys.argv) > 1 else ""
    run_tests(keyword)
