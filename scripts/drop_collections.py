#!/usr/bin/env python
"""Drop one or more ChromaDB collections by name.

This is raw ChromaDB collection-management tooling (list_collections /
delete_collection) with no equivalent in the ``rag.store.RetrievalStore``
abstraction (which manages exactly one already-named collection's chunks, not
the backend's collection catalog) — so unlike the retrieval/indexing engine,
this admin script talks to chromadb directly. It still points at the
configured index_path (RAG_INDEX_PATH in Docker), matching the engine's
connection settings exactly. Lists collections and skips names that don't
exist — safe to re-run.

Usage:
    .venv/bin/python scripts/drop_collections.py obsidian_markdown_bge_small obsidian_markdown_gte_small

Refuses to drop the collection currently named in config.yaml unless --force is
given, so you can't accidentally delete the live index.
"""

import argparse
import sys

from rag.utils import load_config  # sets telemetry env var and patches posthog before chromadb loads

import chromadb


def _get_client(config):
    index_path = config.get("index_path", "./chroma_db")
    return chromadb.PersistentClient(
        path=index_path,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )


def main():
    parser = argparse.ArgumentParser(description="Drop ChromaDB collections by name.")
    parser.add_argument("names", nargs="+", help="Collection name(s) to delete")
    parser.add_argument("--force", action="store_true",
                        help="Allow dropping the collection named in config.yaml")
    args = parser.parse_args()

    config = load_config()
    active = config.get("collection_name", "obsidian_markdown")
    client = _get_client(config)
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
