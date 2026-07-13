#!/usr/bin/env python
"""Drop one or more ChromaDB collections by name.

Reuses the same persistent-client connection logic as the query/index path
(rag.query.get_client), so it points at the configured index_path (RAG_INDEX_PATH
in Docker). Lists collections and skips names that don't exist — safe to re-run.

Usage:
    .venv/bin/python scripts/drop_collections.py obsidian_markdown_bge_small obsidian_markdown_gte_small

Refuses to drop the collection currently named in config.yaml unless --force is
given, so you can't accidentally delete the live index.
"""

import argparse
import sys

from rag.utils import load_config
from rag.query import get_client


def main():
    parser = argparse.ArgumentParser(description="Drop ChromaDB collections by name.")
    parser.add_argument("names", nargs="+", help="Collection name(s) to delete")
    parser.add_argument("--force", action="store_true",
                        help="Allow dropping the collection named in config.yaml")
    args = parser.parse_args()

    config = load_config()
    active = config.get("collection_name", "obsidian_markdown")
    client = get_client(config)
    existing = set(client.list_collections())  # Chroma 0.6.x returns names

    for name in args.names:
        if name == active and not args.force:
            print(f"REFUSING to drop {name!r}: it is the active collection in config.yaml "
                  f"(pass --force to override).")
            continue
        if name not in existing:
            print(f"skip {name!r}: not present")
            continue
        count = client.get_collection(name).count()
        client.delete_collection(name)
        print(f"dropped {name!r} ({count} chunks)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
