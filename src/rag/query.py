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

from .store import get_store
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")

# Cache embedding + reranker models across search() calls in one process. The
# eval harness issues dozens of queries back-to-back; reloading each time is
# wasteful.
_MODEL_CACHE: dict = {}
_RERANKER_CACHE: dict = {}

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_where(domain=None, type_=None, source=None, confidence=None, subdomain=None, status=None):
    """Build a ChromaDB metadata filter from optional field constraints.

    ``status`` is a single scalar in chunk metadata, so it uses a native ``$eq``
    clause here. ``tags`` is deliberately NOT handled here — tags are stored as a
    comma-joined string (Chroma 0.6.3 has weak array support), so tag filtering is
    a post-filter over retrieved records in :func:`search`, never a where clause.
    """
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
    if status:
        filters.append({"status": {"$eq": status}})
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


def open_store(config, collection_name=None):
    """Open the configured ``RetrievalStore`` (or the collection-name override).

    Does not eagerly create anything — for Chroma this defers to
    ``get_collection`` on first use, so querying a collection that has never
    been indexed still raises, exactly like the historical ``open_collection``.
    """
    return get_store(config, collection_name)


def search(
    query,
    n_results=8,
    *,
    filters=None,
    tags=None,
    config=None,
    model=None,
    store=None,
    collection_name=None,
    rerank=False,
    hybrid=False,
):
    """Retrieve the top chunks for ``query``.

    Returns a list of records ordered best-first, each a dict with keys
    ``document`` (str), ``metadata`` (dict), ``distance`` (float, dense L2/cosine
    distance) and ``rank`` (1-based). ``filters`` is a prebuilt ChromaDB
    where-dict (see :func:`build_where`).

    ``model`` / ``store`` may be passed in to avoid reloading them between
    calls; otherwise they are resolved from ``config`` (loaded if omitted).
    ``rerank`` retrieves a wider dense pool (``rerank_fetch_k``, default 20) and
    reorders it to the top ``n_results`` with a cross-encoder. ``hybrid`` (BM25
    fusion) is Phase 3d — accepted but not yet wired.

    ``tags`` is a list of tag names applied as a post-filter (exact, case-
    insensitive membership; multiple tags = AND) — Chroma can't filter the
    comma-joined ``tags`` metadata string natively. When set, the dense pool is
    widened to ``tag_fetch_k`` (default 200) so the post-filter has candidates to
    keep; filtering runs before rerank/trim, so a very rare tag may still
    under-return within that pool (best-effort).
    """
    if config is None:
        config = load_config()
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if model is None:
        model = get_model(model_name)
    if store is None:
        store = open_store(config, collection_name)

    # Config-driven query prefix (e.g. bge's retrieval instruction). Empty for
    # models that need none (MiniLM, gte); the passage/index side never prefixes.
    query_instruction = config.get("query_instruction", "")
    embed_input = f"{query_instruction}{query}"
    query_embedding = model.encode([embed_input], normalize_embeddings=True).tolist()[0]

    # When reranking, retrieve a wider dense candidate pool (default 20) and let
    # the cross-encoder pick the final n_results from it. (hybrid is Phase 3d.)
    rerank_fetch_k = int(config.get("rerank_fetch_k", 20))
    fetch_k = max(n_results, rerank_fetch_k) if rerank else n_results
    # Tags are post-filtered (not a native where clause), so widen the dense pool
    # to give the post-filter enough candidates to keep. Best-effort: a very rare
    # tag may still under-return within this pool.
    if tags:
        fetch_k = max(fetch_k, int(config.get("tag_fetch_k", 200)))

    hits = store.query(query_embedding, fetch_k, filters)

    records = [
        {"document": hit["document"], "metadata": hit["metadata"], "distance": hit["distance"], "rank": i}
        for i, hit in enumerate(hits, start=1)
    ]

    # Tag post-filter (before rerank so the cross-encoder scores the filtered
    # pool). Tags live as a comma-joined metadata string; keep a record only if
    # every requested tag is an exact member of its tag set (case-insensitive) —
    # so "ci" must not match "ci-cd", and multiple tags are AND (subset test).
    # Records with no/empty tags metadata never raise and are dropped.
    if tags:
        want = {t.strip().lower() for t in tags if t and t.strip()}
        if want:
            records = [
                r for r in records
                if want <= {
                    s.strip().lower()
                    for s in (r["metadata"].get("tags") or "").split(",")
                    if s.strip()
                }
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

    logger.info("query=%r n=%d filter=%s tags=%s rerank=%s -> %d results",
                query, n_results, filters, tags, rerank, len(records))
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
  rag-query "kubernetes" --tag devops -n 5
  rag-query "deployment" --status processed
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
    parser.add_argument("--status", default=None,
                        help="Filter by status metadata (native $eq, e.g. processed)")
    parser.add_argument("--tag", action="append", default=None, metavar="TAG",
                        help="Keep only chunks carrying this tag (exact match, "
                             "case-insensitive). Repeatable; multiple --tag = AND.")
    parser.add_argument("--json", dest="output_json", action="store_true",
                        help="Output results as a JSON array")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Disable cross-encoder reranking (dense retrieval only)")
    parser.set_defaults(rerank=True)
    args = parser.parse_args()

    query = " ".join(args.query)
    config = load_config()
    setup_logging(config, console=False)  # log to file only; results print to stdout

    where = build_where(args.domain, args.type_, args.source, args.confidence, args.subdomain,
                        status=args.status)
    records = search(query, args.n_results, filters=where, tags=args.tag, config=config,
                     rerank=args.rerank)

    if args.output_json:
        output = [
            {"distance": r["distance"], "document": r["document"], **r["metadata"]}
            for r in records
        ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print()
    print("Query: " + query)
    if where or args.tag:
        # `where` already carries status (via build_where); tags are a separate
        # post-filter, so print them explicitly alongside the where-dict.
        parts = []
        if where:
            parts.append(json.dumps(where))
        if args.tag:
            parts.append("tags=" + json.dumps(args.tag))
        print("Filter: " + "  ".join(parts))
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
