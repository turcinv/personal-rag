"""Three-way metric comparison CLI.

Runs the same query against three ChromaDB collections (L2, cosine, dot product)
and prints results side-by-side ranked by position, so it is easy to see how
retrieval differs across distance metrics.

Entry point: rag-compare
"""

import argparse
import logging

from .utils import load_config, setup_logging
from .query import build_where

import chromadb
from sentence_transformers import SentenceTransformer

# Fixed collection names for the three metrics.
_METRICS = [
    ("L2",  "obsidian_markdown"),
    ("COS", "obsidian_markdown_cos"),
    ("DOT", "obsidian_markdown_dot"),
]

# ── ANSI colours ──────────────────────────────────────────────────────────────

import sys

def _ansi(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

BOLD  = lambda t: _ansi("1",  t)
DIM   = lambda t: _ansi("2",  t)
CYAN  = lambda t: _ansi("36", t)
GREEN = lambda t: _ansi("32", t)
LABEL_COLOR = {"L2": lambda t: _ansi("33", t),   # yellow
               "COS": lambda t: _ansi("32", t),   # green
               "DOT": lambda t: _ansi("36", t)}   # cyan


# ── Query helpers ─────────────────────────────────────────────────────────────

def _query_collection(client: chromadb.PersistentClient, name: str,
                      embedding: list, n: int, where: dict | None) -> list[dict]:
    """Query a single collection; returns [] if the collection does not exist."""
    try:
        col = client.get_collection(name)
    except Exception:
        return []
    kwargs = dict(
        query_embeddings=[embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    return [
        {"rank": i + 1, "distance": d, "title": m.get("title", ""), "heading": m.get("heading", ""),
         "path": m.get("path", ""), "domain": m.get("domain", ""), "subdomain": m.get("subdomain", ""),
         "type": m.get("type", ""), "document": doc}
        for i, (d, m, doc) in enumerate(zip(res["distances"][0], res["metadatas"][0], res["documents"][0]))
    ]


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt_result(label: str, r: dict | None) -> str:
    color = LABEL_COLOR.get(label, lambda t: t)
    if r is None:
        return f"  {color(f'[{label}]')}  {DIM('— no result —')}"
    dist = r["distance"]
    title = r["title"] or r["path"].split("/")[-1]
    heading = r["heading"]
    name = f"{title} — {heading}" if heading and heading.lower() != "summary" else title
    sub = r.get("subdomain", "")
    domain = f"{r['domain']}" + (f" / {sub}" if sub else "")
    path_short = r["path"]
    return (
        f"  {color(f'[{label}]')}  {dist:.4f}  {BOLD(name[:60])}\n"
        f"          {DIM(domain[:50])}  {DIM(path_short[:70])}"
    )


def _print_comparison(query: str, results_by_metric: dict[str, list[dict]],
                      n: int, where: dict | None) -> None:
    sep = "─" * 90
    print()
    print(BOLD(f"Query: {query}"))
    if where:
        print(DIM(f"Filter: {where}"))
    print()

    # Check which collections were actually available
    missing = [label for label, res in results_by_metric.items() if not res]
    if missing:
        print(DIM(f"  Collections not found (not yet indexed?): {', '.join(missing)}"))
        print()

    for rank in range(1, n + 1):
        print(sep)
        print(BOLD(f"  Rank {rank}"))
        for label, _ in _METRICS:
            res_list = results_by_metric.get(label, [])
            r = res_list[rank - 1] if len(res_list) >= rank else None
            print(_fmt_result(label, r))
        print()

    print(sep)
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare query results across L2, cosine, and dot-product collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  rag-compare "What do I know about Kubernetes?"
  rag-compare "RAG and vector search" -n 5
  rag-compare "Kubernetes" --domain DevOps
""",
    )
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("-n", "--n-results", type=int, default=5, metavar="N",
                        help="Number of results per metric (default: 5)")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--subdomain", default=None)
    parser.add_argument("--type", dest="type_", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--confidence", default=None)
    args = parser.parse_args()

    query = " ".join(args.query)
    config = load_config()
    setup_logging(config, console=False)
    logger = logging.getLogger("rag")

    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    index_path = config.get("index_path", "./chroma_db")

    model = SentenceTransformer(model_name)
    embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )

    where = build_where(args.domain, args.type_, args.source, args.confidence, args.subdomain)

    results_by_metric: dict[str, list[dict]] = {}
    for label, coll_name in _METRICS:
        results_by_metric[label] = _query_collection(client, coll_name, embedding, args.n_results, where)
        logger.info("compare metric=%s collection=%s n=%d filter=%s -> %d results",
                    label, coll_name, args.n_results, where, len(results_by_metric[label]))

    _print_comparison(query, results_by_metric, args.n_results, where)


if __name__ == "__main__":
    main()
