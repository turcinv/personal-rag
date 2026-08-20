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

from .provenance import ensure_compatible, expected_provenance, read_index_state
from .retrieval import RetrievalFilter
from .store import compile_where, get_store
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("rag")

# Cache embedding + reranker models across search() calls in one process. The
# eval harness issues dozens of queries back-to-back; reloading each time is
# wasteful.
_MODEL_CACHE: dict = {}
_RERANKER_CACHE: dict = {}

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_default(config: dict) -> bool:
    """Whether reranking is on by default for this config profile.

    ``search()`` itself always defaults to ``rerank=False``; this resolves what
    the *callers* (``rag-query``, ``make eval``, ``POST /query``, ``POST /answer``)
    should do when the caller did not say either way. It is per-profile because
    the cross-encoder helps or hurts depending on the corpus: on the personal KB
    it costs overall recall@5 (0.911 → 0.844 on the golden set — it reshuffles
    inside top-5 but ejects some rank-1 hits), so that profile sets it False.
    Falls back to True when the key is absent, preserving pre-2026-07-27 behavior
    for configs that predate it.
    """
    return bool((config or {}).get("rerank_default", True))


def build_where(domain=None, type_=None, source=None, confidence=None, subdomain=None, status=None):
    """Compile optional field constraints into a Chroma where-dict.

    LEGACY/compat shim: shipped callers now build a backend-neutral
    :class:`~rag.retrieval.RetrievalFilter` and let the store compile it. This
    function is kept for backward compatibility and delegates to the single
    Chroma-syntax compiler (``rag.store.compile_where``) so the ``$eq``/``$and``
    shape is defined in exactly one place. ``tags`` are never a where clause —
    they are post-filtered in :func:`search`.
    """
    return compile_where(
        RetrievalFilter.create(
            domain=domain,
            subdomain=subdomain,
            type=type_,
            source=source,
            confidence=confidence,
            status=status,
        )
    )


def get_model(model_name, revision=None):
    """Return a cached SentenceTransformer for ``model_name`` (optionally pinned).

    ``revision`` pins the exact model snapshot (a HF commit/tag) so an offline or
    reproducible deployment always loads the same weights; it participates in the
    cache key so two revisions never collide.
    """
    key = (model_name, revision)
    if key not in _MODEL_CACHE:
        if revision:
            _MODEL_CACHE[key] = SentenceTransformer(model_name, revision=revision)
        else:
            _MODEL_CACHE[key] = SentenceTransformer(model_name)
    return _MODEL_CACHE[key]


def get_reranker(model_name):
    """Return a cached CrossEncoder reranker for ``model_name`` (lazy import —
    only pulled in when reranking is actually requested)."""
    if model_name not in _RERANKER_CACHE:
        from sentence_transformers import CrossEncoder
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def open_store(config, collection_name=None, model=None):
    """Open a store and validate it against the active embedding profile."""
    store = get_store(config, collection_name)
    if model is not None:
        name = collection_name or config.get("collection_name", "obsidian_markdown")
        ensure_compatible(
            store,
            expected_provenance(config, model=model, collection_name=name),
        )
    return store


def search(
    query,
    n_results=8,
    *,
    retrieval_filter=None,
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

    ``retrieval_filter`` is the backend-neutral
    :class:`~rag.retrieval.RetrievalFilter` shipped callers build; the store
    compiles it to its own dialect and its ``tags`` drive the post-filter.
    ``filters`` (a raw where-dict) and ``tags`` (a list) are the legacy
    equivalents kept for the retrieval-algorithm tests and pre-DTO callers;
    ``retrieval_filter`` takes precedence when given.

    ``model`` / ``store`` may be passed in to avoid reloading them between
    calls; otherwise they are resolved from ``config`` (loaded if omitted).
    ``rerank`` retrieves a wider dense pool (``rerank_fetch_k``, default 20) and
    reorders it to the top ``n_results`` with a cross-encoder. ``hybrid`` fuses a
    BM25 lexical pool with the dense pool via Reciprocal Rank Fusion: for a store
    with native fusion (``supports_hybrid``) the store does it; for Chroma it is
    done client-side here against ``rag.lexical`` (which needs ``make build-lexical``
    first). ``hybrid`` composes with ``rerank`` (fuse, then rerank the fused pool)
    and is off by default — ``hybrid=False`` is byte-identical to dense-only.

    ``tags`` is a list of tag names applied as a post-filter (exact, case-
    insensitive membership; multiple tags = AND) — Chroma can't filter the
    comma-joined ``tags`` metadata string natively. When set, the dense pool is
    widened to ``tag_fetch_k`` (default 200) so the post-filter has candidates to
    keep; filtering runs before rerank/trim, so a very rare tag may still
    under-return within that pool (best-effort).
    """
    # Resolve the active filter + tag list from either the neutral DTO
    # (preferred) or the legacy filters/tags kwargs. An empty DTO collapses to
    # None so the store/lexical widening stays byte-identical to "no filter".
    if retrieval_filter is not None:
        where = None if retrieval_filter.is_empty else retrieval_filter
        tag_list = list(retrieval_filter.tags)
    else:
        where = filters
        tag_list = tags

    if config is None:
        config = load_config()
    model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    if model is None:
        model = get_model(model_name, revision=config.get("embedding_revision") or None)
    if store is None:
        store = open_store(config, collection_name, model=model)

    # Config-driven query prefix (e.g. bge's retrieval instruction). Empty for
    # models that need none (MiniLM, gte); the passage/index side never prefixes.
    query_instruction = config.get("query_instruction", "")
    embed_input = f"{query_instruction}{query}"
    query_embedding = model.encode([embed_input], normalize_embeddings=True).tolist()[0]

    # When reranking, retrieve a wider dense candidate pool (default 20) and let
    # the cross-encoder pick the final n_results from it.
    rerank_fetch_k = int(config.get("rerank_fetch_k", 20))
    fetch_k = max(n_results, rerank_fetch_k) if rerank else n_results
    # Tags are post-filtered (not a native where clause), so widen the dense pool
    # to give the post-filter enough candidates to keep. Best-effort: a very rare
    # tag may still under-return within this pool.
    if tag_list:
        fetch_k = max(fetch_k, int(config.get("tag_fetch_k", 200)))
    # Hybrid widens BOTH the dense and lexical pools independent of n_results, so
    # RRF has depth to rescue a dense miss with a lexical hit (and vice-versa).
    if hybrid:
        fetch_k = max(fetch_k, int(config.get("hybrid_fetch_k", 50)))

    # Thread the raw query text + hybrid flag down to the store. Chroma ignores
    # both (pure vector search); a backend with native BM25+k-NN fusion uses
    # them. `text` is the raw query — never `embed_input` (the prefixed variant).
    hits = store.query(query_embedding, fetch_k, where, text=query, hybrid=hybrid)

    # Client-side BM25 fusion: only when hybrid is requested AND the store has no
    # native lexical channel (Chroma). A native-fusion store (supports_hybrid)
    # already returned fused order — trust it. hybrid=False never enters here, so
    # the dense-only path stays byte-identical.
    if hybrid and not getattr(store, "supports_hybrid", False):
        from .lexical import get_lexical, rrf_fuse  # lazy: only on the hybrid path
        state = read_index_state(store)
        if state is None:
            raise RuntimeError(
                "Hybrid retrieval requires vector generation metadata; rebuild or "
                "explicitly adopt index provenance, then run make build-lexical."
            )
        lex_hits = get_lexical(
            config, collection_name, index_state=state
        ).query(query, fetch_k, where=where)
        hits = rrf_fuse(
            hits, lex_hits,
            weights=config.get("hybrid_weights", (1.0, 1.0)),
            k_rrf=int(config.get("hybrid_rrf_k", 60)),
        )

    records = [
        {"document": hit["document"], "metadata": hit["metadata"], "distance": hit["distance"], "rank": i}
        for i, hit in enumerate(hits, start=1)
    ]

    # Tag post-filter (before rerank so the cross-encoder scores the filtered
    # pool). Tags live as a comma-joined metadata string; keep a record only if
    # every requested tag is an exact member of its tag set (case-insensitive) —
    # so "ci" must not match "ci-cd", and multiple tags are AND (subset test).
    # Records with no/empty tags metadata never raise and are dropped.
    if tag_list:
        want = {t.strip().lower() for t in tag_list if t and t.strip()}
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

    # Privacy: the raw query text is NOT logged by default (it fans out to a
    # text file + SQLite). Set `log_queries: true` in config to log it verbatim
    # for debugging; otherwise only its length is recorded.
    if config.get("log_queries"):
        logger.info("query=%r n=%d filter=%s tags=%s rerank=%s -> %d results",
                    query, n_results, where, tag_list, rerank, len(records))
    else:
        logger.info("query=<redacted len=%d> n=%d filter=%s tags=%s rerank=%s -> %d results",
                    len(query or ""), n_results, bool(where), bool(tag_list),
                    rerank, len(records))
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
    rr = parser.add_mutually_exclusive_group()
    rr.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="Force cross-encoder reranking on (overrides rerank_default)")
    rr.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="Disable cross-encoder reranking (dense retrieval only)")
    parser.add_argument("--hybrid", dest="hybrid", action="store_true", default=False,
                        help="Fuse BM25 lexical retrieval with dense via RRF "
                             "(needs `make build-lexical` first)")
    args = parser.parse_args()

    query = " ".join(args.query)
    config = load_config()
    setup_logging(config, console=False)  # log to file only; results print to stdout

    # Neither flag given → fall back to the profile's rerank_default.
    rerank = args.rerank if args.rerank is not None else rerank_default(config)

    retrieval_filter = RetrievalFilter.create(
        domain=args.domain,
        subdomain=args.subdomain,
        type=args.type_,
        source=args.source,
        confidence=args.confidence,
        status=args.status,
        tags=args.tag,
    )
    records = search(query, args.n_results, retrieval_filter=retrieval_filter,
                     config=config, rerank=rerank, hybrid=args.hybrid)

    if args.output_json:
        output = [
            {"distance": r["distance"], "document": r["document"], **r["metadata"]}
            for r in records
        ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print()
    print("Query: " + query)
    if not retrieval_filter.is_empty:
        # Neutral, backend-agnostic display — the CLI never prints store syntax.
        parts = []
        scalars = retrieval_filter.scalar_constraints()
        if scalars:
            parts.append(", ".join(f"{name}={value}" for name, value in scalars))
        if retrieval_filter.tags:
            parts.append("tags=" + json.dumps(list(retrieval_filter.tags)))
        print("Filter: " + "  ".join(parts))
    print()

    for i, r in enumerate(records, start=1):
        doc, meta, distance = r["document"], r["metadata"], r["distance"]
        print("=" * 80)
        print(f"{i}. {meta.get('title')} - {meta.get('heading')}")
        print(f"Path: {meta.get('path')}")
        _sub = meta.get('subdomain')
        print(f"Type: {meta.get('type')} | Domain: {meta.get('domain')}" + (f" / {_sub}" if _sub else "") + f" | Status: {meta.get('status')} | Confidence: {meta.get('confidence')}")
        # distance is None for a lexical-only hybrid hit (no cosine distance).
        print("Distance: " + (f"{distance:.4f}" if distance is not None else "n/a"))
        print("-" * 80)
        print(doc[:1200].strip())
        print()


if __name__ == "__main__":
    main()
