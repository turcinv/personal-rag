"""Semantic query CLI with metadata filters and JSON output."""

import argparse
import json
import logging

from .utils import load_config, setup_logging  # sets telemetry env var and patches posthog before chromadb loads

import chromadb
from sentence_transformers import SentenceTransformer


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
    args = parser.parse_args()

    query = " ".join(args.query)
    config = load_config()
    setup_logging(config, console=False)  # log to file only; results print to stdout
    logger = logging.getLogger("rag")
    model_name      = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
    index_path      = config.get("index_path", "./chroma_db")
    collection_name = config.get("collection_name", "obsidian_markdown")

    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    client = chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(collection_name)

    where = build_where(args.domain, args.type_, args.source, args.confidence, args.subdomain)
    query_kwargs = dict(
        query_embeddings=[query_embedding],
        n_results=args.n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    logger.info("query=%r n=%d filter=%s -> %d results", query, args.n_results, where, len(docs))

    if args.output_json:
        output = [
            {"distance": dist, "document": doc, **meta}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    print()
    print("Query: " + query)
    if where:
        print("Filter: " + json.dumps(where))
    print()

    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
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
