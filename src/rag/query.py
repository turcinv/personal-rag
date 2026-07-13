"""Semantic query over the indexed vault + PDFs.

The retrieval core lives in :func:`search` — an importable seam shared by this
CLI (``rag-query``) and the eval harness (``scripts/eval_recall.py``). ``main()``
is a thin wrapper that parses args, builds a metadata filter, calls ``search()``,
and formats the output. Reranking (Phase 3c) and BM25 hybrid (Phase 3d) hook in
through the ``rerank`` / ``hybrid`` toggles on ``search()``.
"""

import argparse
import json
import logging

from .utils import load_config, setup_logging  # sets telemetry env var and patches posthog before chromadb loads

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")

# Cache embedding + reranker models across search() calls in one process. The
# eval harness issues dozens of queries back-to-back; reloading each time is
# wasteful.
_MODEL_CACHE: dict = {}
_RERANKER_CACHE: dict = {}

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_where(domain=None, type_=None, source=None, confidence=None, subdomain=None):
    """Build a ChromaDB metadata filter from optional field constraints."""
    filters = []
    if domain:
        filters.append({"domain": {"$eq": domain}})
    if subdomain:
        filters.append({"subdomain": {"$eq": subdomain}})
    if type_:
        filters.append({"type": {"$eq": type_}})
    if source:
        filters.append({"source": {"$eq": source}})
    if confidence:
        filters.append({"confidence": {"$eq": confidence}})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def get_model(model_name):
    """Return a cached SentenceTransformer for ``model_name``."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def get_reranker(model_name):
    """Return a cached CrossEncoder reranker for ``model_name`` (lazy import —
    only pulled in when reranking is actually requested)."""
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def get_client(config):
    """Open the persistent ChromaDB client at the configured index path."""
    index_path = config.get("index_path", "./chroma_db")
    return chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def open_collection(config, collection_name=None):
    """Open the persistent ChromaDB collection named in config (or the override)."""
    name = collection_name or config.get("collection_name", "obsidian_markdown")
    return get_client(config).get_collection(name)


def search(
    query,
    n_results=8,
    *,
    filters=None,
    config=None,
    model=None,
    collection=None,
    collection_name=None,
    rerank=False,
    hybrid=False,
):
    """Retrieve the top chunks for ``query``.

    Returns a list of records ordered best-first, each a dict with keys
    ``document`` (str), ``metadata`` (dict), ``distance`` (float, dense L2/cosine
    distance) and ``rank`` (1-based). ``filters`` is a prebuilt ChromaDB
    where-dict (see :func:`build_where`).

    ``model`` / ``collection`` may be passed in to avoid reloading them between
    calls; otherwise they are resolved from ``config`` (loaded if omitted).
    ``rerank`` retrieves a wider dense pool (``rerank_fetch_k``, default 20) and
    reorders it to the top ``n_results`` with a cross-encoder. ``hybrid`` (BM25
    fusion) is Phase 3d — accepted but not yet wired.
    """
    if config is None:
        config = load_config()
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if model is None:
        model = get_model(model_name)
    if collection is None:
        collection = open_collection(config, collection_name)

    # Config-driven query prefix (e.g. bge's retrieval instruction). Empty for
    # models that need none (MiniLM, gte); the passage/index side never prefixes.
    query_instruction = config.get("query_instruction", "")
    embed_input = f"{query_instruction}{query}"
    query_embedding = model.encode([embed_input], normalize_embeddings=True).tolist()[0]

    # When reranking, retrieve a wider dense candidate pool (default 20) and let
    # the cross-encoder pick the final n_results from it. (hybrid is Phase 3d.)
    rerank_fetch_k = int(config.get("rerank_fetch_k", 20))
    fetch_k = max(n_results, rerank_fetch_k) if rerank else n_results

    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    if filters:
        query_kwargs["where"] = filters

    results = collection.query(**query_kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    records = [
        {"document": doc, "metadata": meta, "distance": dist, "rank": i}
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1)
    ]

    # Cross-encoder rerank: score each (raw query, chunk) pair and reorder, then
    # trim to n_results. The reranker sees the natural query — never the
    # embedding-side instruction prefix.
    if rerank and records:
        reranker = get_reranker(config.get("reranker_model", DEFAULT_RERANKER))
        scores = reranker.predict([(query, r["document"]) for r in records])
        ranked = sorted(zip(records, scores), key=lambda rs: rs[1], reverse=True)
        records = []
        for new_rank, (rec, score) in enumerate(ranked[:n_results], start=1):
            rec["rerank_score"] = float(score)
            rec["rank"] = new_rank
            records.append(rec)
    else:
        records = records[:n_results]

    logger.info("query=%r n=%d filter=%s rerank=%s -> %d results",
                query, n_results, filters, rerank, len(records))
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Semantic query over indexed Obsidian vault and PDFs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  rag-query "What do I know about Kubernetes?"
  rag-query "secrets in Python" -n 12
  rag-query "Kubernetes" --domain DevOps
  rag-query "testing" --domain "Software Engineering" --subdomain "Python & Backend Development"
  rag-query "deployment" --domain DevOps --confidence high
  rag-query "book recommendations" --source pdf --type book
  rag-query "RAG pipeline" --json
""",
    )
    parser.add_argument("query", nargs="+", help="Query text")
    parser.add_argument("-n", "--n-results", type=int, default=8, metavar="N",
                        help="Number of results to return (default: 8)")
    parser.add_argument("--domain", default=None,
                        help="Filter by domain metadata (e.g. DevOps, 'Software Engineering')")
    parser.add_argument("--subdomain", default=None,
                        help="Filter by subdomain metadata (subfolder, e.g. 'Python & Backend Development')")
    parser.add_argument("--type", dest="type_", default=None,
                        help="Filter by type metadata (e.g. book, resource, Knowledge)")
    parser.add_argument("--source", default=None,
                        help="Filter by source metadata (e.g. pdf)")
    parser.add_argument("--confidence", default=None,
                        help="Filter by confidence metadata (e.g. high, medium)")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Output results as a JSON array")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Disable cross-encoder reranking (dense retrieval only)")
    parser.set_defaults(rerank=True)
    args = parser.parse_args()

    query = " ".join(args.query)
    config = load_config()
    setup_logging(config, console=False)  # log to file only; results print to stdout

    where = build_where(args.domain, args.type_, args.source, args.confidence, args.subdomain)
    records = search(query, args.n_results, filters=where, config=config, rerank=args.rerank)

    if args.output_json:
        output = [
            {"distance": r["distance"], "document": r["document"], **r["metadata"]}
            for r in records
        ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print()
    print("Query: " + query)
    if where:
        print("Filter: " + json.dumps(where))
    print()

    for i, r in enumerate(records, start=1):
        doc, meta, distance = r["document"], r["metadata"], r["distance"]
        print("=" * 80)
        print(f"{i}. {meta.get('title')} - {meta.get('heading')}")
        print(f"Path: {meta.get('path')}")
        _sub = meta.get('subdomain')
        print(f"Type: {meta.get('type')} | Domain: {meta.get('domain')}" + (f" / {_sub}" if _sub else "") + f" | Status: {meta.get('status')} | Confidence: {meta.get('confidence')}")
        print(f"Distance: {distance:.4f}")
        print("-" * 80)
        print(doc[:1200].strip())
        print()


if __name__ == "__main__":
    main()
