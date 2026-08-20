"""Index statistics CLI.

Prints total chunk count, per-source-type breakdown (chunk count + unique
file count), and optionally lists all collections in the configured index.
Uses paginated iteration via ``store.iter_records(page_size)`` so it never
materialises the full index in memory — safe on Jetson (8 GB unified RAM).

Entry point: rag-stats
"""

import argparse

from .utils import load_config, setup_logging
from .store import get_store, list_collection_names


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _infer_source_type(meta: dict) -> str:
    """Infer whether a chunk came from the vault or from a book/resource.

    The extractors don't store a literal ``source_type`` field. Vault markdown
    chunks have a relative path with ``/`` separators (e.g.
    ``Knowledge/DevOps/Docker.md``) and their ``source`` metadata is typically
    empty. Book/resource chunks (from JSON or live PDF) store only the filename
    in ``path`` (no ``/``) and set ``source`` to ``"pdf"``.
    """
    path = meta.get("path", "")
    source = meta.get("source", "")

    if "/" in path:
        return "markdown"
    if source == "pdf" or path.endswith(".pdf"):
        return "book/resource"
    return "other"


def main():
    parser = argparse.ArgumentParser(
        prog="rag-stats",
        description="Print index statistics (chunk counts, source breakdown, collections).",
    )
    parser.add_argument(
        "--collections", action="store_true",
        help="List all collection names in the index and exit",
    )
    parser.add_argument(
        "--page-size", type=_positive_int, default=10_000,
        help="Iteration page size for metadata aggregation (default: 10000)",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config, console=False)

    if args.collections:
        names = list_collection_names(config)
        print(f"Collections ({len(names)}):")
        for name in sorted(names):
            print(f"  {name}")
        return

    store = get_store(config)
    total = store.count()
    print(f"Collection: {store.name}")
    print(f"Total chunks: {total:,}")

    # Paginated aggregation — never materialises full index
    type_counts = {}   # type: dict[str, int]
    type_files = {}    # type: dict[str, set[str]]

    for _id, _doc, meta in store.iter_records(page_size=args.page_size):
        st = _infer_source_type(meta)
        type_counts[st] = type_counts.get(st, 0) + 1
        path = meta.get("path", "")
        if path:
            type_files.setdefault(st, set()).add(path)

    if type_counts:
        print(f"\nBy source type:")
        for st in sorted(type_counts):
            count = type_counts[st]
            files = len(type_files.get(st, set()))
            print(f"  {st:12s}  {count:>7,} chunks  ({files:,} files)")
