"""Client-side BM25 lexical index (SQLite FTS5) + Reciprocal Rank Fusion.

This is the Chroma-backend fallback for hybrid retrieval. ``ChromaStore`` has no
BM25 channel (``supports_hybrid = False``), so when a caller asks ``search()`` for
``hybrid=True`` we retrieve a lexical candidate pool from an FTS5 index built over
*the same chunks already in the vector store* and fuse it with the dense pool via
RRF — entirely above the store, keeping ``chromadb`` confined to
``store/chroma_store.py``.

The lexical index is a build artifact (like ``chroma_db/``): built by
``rag-build-lexical`` (``make build-lexical``) from ``store.iter_records()``, so it
needs no re-chunking or re-embedding, and persisted to
``./lexical_index/<collection_name>.db`` (gitignored). See CLAUDE.md roadmap item 6
and docs/OPENSEARCHSTORE_IMPLEMENTATION_PLAN.md (this is the Chroma-only stand-in
for the native OpenSearch ``hybrid=True`` path).

Design notes:
  - FTS5 is stdlib (no new dependency) and already proven on every target by
    ``src/extractor/build_sqlite.py``. We do NOT reuse that ``resources.db``: it is
    whole-document grain keyed by a path-only hash with a different membership set,
    so its IDs never coincide with the content-hash *chunk* IDs used here.
  - ``bm25()`` returns lower-is-better (best match most negative), so ``ORDER BY
    score`` ascending is correct.
  - A lexical hit carries ``distance = None`` (a sentinel) — it has no cosine
    distance, and ``search()`` trusts store/fused order rather than re-sorting on it.
"""

import argparse
import json
import logging
import os
import re
import sqlite3

from .utils import load_config, setup_logging
from .store import get_store

logger = logging.getLogger("rag")

# One row per Chroma chunk. `document` is the only analyzed column; `chunk_id`
# and `metadata` (a JSON blob) ride along UNINDEXED. porter stems English
# (troubleshoot↔troubleshooting); unicode61 folds diacritics.
_SCHEMA = """
CREATE VIRTUAL TABLE chunks USING fts5(
    chunk_id UNINDEXED,
    document,
    metadata UNINDEXED,
    tokenize = 'porter unicode61'
);
"""

# Unicode-aware word tokens. `\w` on a `str` is Unicode by default in Python 3,
# so "nasazení" stays one token; re.UNICODE is explicit-but-redundant.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Reuse one read connection per db path across the eval's 45 queries (mirrors
# query._MODEL_CACHE). Populated lazily, only on the hybrid path.
_LEXICAL_CACHE: dict = {}


def _to_match_query(text):
    """Turn a natural-language query into a safe FTS5 MATCH expression.

    Extracts word tokens, double-quotes each (which neutralizes FTS5 operator
    keywords like AND/OR/NOT/NEAR and every special char — a quoted `\\w+` run
    can never break the syntax), and OR-joins them so bm25 rewards documents
    matching more/rarer terms. Returns ``None`` when there are zero tokens
    (empty / pure-punctuation / emoji query) — the caller must then skip MATCH
    entirely, because ``MATCH ''`` raises ``OperationalError``.
    """
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


def _matches_where(meta, where):
    """Evaluate a Chroma-style where-dict against a chunk's metadata.

    Handles exactly the shapes ``query.build_where`` emits: a single
    ``{field: {"$eq": v}}`` clause or ``{"$and": [clauses...]}``. Used only to
    post-filter lexical candidates for ``rag-query --hybrid --domain X`` — the
    golden-set eval is unfiltered, so this never fires there.
    """
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    for field, cond in where.items():
        want = cond["$eq"] if isinstance(cond, dict) and "$eq" in cond else cond
        if meta.get(field) != want:
            return False
    return True


class LexicalIndex:
    """Read-only BM25 view over one FTS5 lexical index file."""

    def __init__(self, path):
        self.path = os.fspath(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        # check_same_thread=False: read-only queries, safe to share if a future
        # threaded caller reuses the cached instance.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)

    @staticmethod
    def build(records, path, batch_size=5000):
        """Build (or rebuild) the FTS5 index at ``path`` from ``records``.

        ``records`` is an iterable of ``(chunk_id, document, metadata)`` —
        typically ``store.iter_records()``. Drops any existing db first (like
        ``build_sqlite.py``) and inserts in batched transactions. Returns the
        number of rows written.
        """
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        for suffix in ("", "-wal", "-shm"):
            stale = path + suffix
            if os.path.exists(stale):
                os.remove(stale)

        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            cur = conn.cursor()
            n = 0
            batch = []
            for cid, doc, meta in records:
                batch.append((cid, doc or "", json.dumps(meta or {}, ensure_ascii=False)))
                if len(batch) >= batch_size:
                    cur.executemany(
                        "INSERT INTO chunks(chunk_id, document, metadata) VALUES (?, ?, ?)",
                        batch,
                    )
                    n += len(batch)
                    batch.clear()
            if batch:
                cur.executemany(
                    "INSERT INTO chunks(chunk_id, document, metadata) VALUES (?, ?, ?)",
                    batch,
                )
                n += len(batch)
            conn.commit()
            return n
        finally:
            conn.close()

    def query(self, text, k, where=None):
        """Return up to ``k`` BM25 records for ``text``, best-first.

        Each record is ``{"document", "metadata", "distance": None}`` — the same
        shape ``store.query()`` returns, minus a real distance. Returns ``[]``
        for a query with no word tokens (never executes ``MATCH ''``). When
        ``where`` is given, candidates are post-filtered in Python; the internal
        fetch is widened so the post-filter has room (best-effort, same accepted
        trade-off as the tag post-filter in ``query.search()``).
        """
        match = _to_match_query(text)
        if match is None:
            return []
        fetch = k if where is None else max(k, 500)
        rows = self._conn.execute(
            "SELECT document, metadata FROM chunks WHERE chunks MATCH ? "
            "ORDER BY bm25(chunks) LIMIT ?",
            (match, fetch),
        ).fetchall()

        out = []
        for doc, meta_json in rows:
            meta = json.loads(meta_json) if meta_json else {}
            if where is not None and not _matches_where(meta, where):
                continue
            out.append({"document": doc, "metadata": meta, "distance": None})
            if len(out) >= k:
                break
        return out


def rrf_fuse(dense, lexical, *, weights=(1.0, 1.0), k_rrf=60):
    """Reciprocal Rank Fusion of a dense and a lexical result list.

    Keyed by document text (dense records carry no chunk id, and text is a safe
    unique key because chunk IDs are content hashes — identical text collapses to
    one chunk). ``weights`` is ``(w_lexical, w_dense)`` matching the config
    ``hybrid_weights: [w_lexical, w_dense]`` order. Score for a document:

        w_dense / (k_rrf + rank_dense)  +  w_lexical / (k_rrf + rank_lexical)

    (a list the document is absent from contributes 0). Ranks are 1-based within
    each list. When a document appears in both lists the *dense* record is kept
    (it has a real cosine distance). The result is ordered by a fully
    deterministic key — ``(-score, dense_rank, lexical_rank, document)`` with a
    missing rank treated as +inf — so ties never leak Python's sort stability /
    pool-iteration order into the output.
    """
    w_lex, w_dense = weights
    inf = float("inf")
    record = {}
    dense_rank = {}
    lexical_rank = {}

    for i, hit in enumerate(dense, start=1):
        doc = hit["document"]
        dense_rank.setdefault(doc, i)
        record.setdefault(doc, hit)          # prefer the dense record
    for i, hit in enumerate(lexical, start=1):
        doc = hit["document"]
        lexical_rank.setdefault(doc, i)
        record.setdefault(doc, hit)          # only fills docs dense didn't have

    def score(doc):
        s = 0.0
        if doc in dense_rank:
            s += w_dense / (k_rrf + dense_rank[doc])
        if doc in lexical_rank:
            s += w_lex / (k_rrf + lexical_rank[doc])
        return s

    ordered = sorted(
        record,
        key=lambda doc: (-score(doc), dense_rank.get(doc, inf),
                         lexical_rank.get(doc, inf), doc),
    )
    return [record[doc] for doc in ordered]


def lexical_path_for(config, collection_name=None):
    """Resolve the lexical index path for a config profile.

    Explicit ``lexical_path`` in config wins; otherwise it defaults to
    ``./lexical_index/<collection_name>.db`` so profiles (personal vs logmanager)
    auto-isolate by collection name.
    """
    explicit = (config or {}).get("lexical_path")
    if explicit:
        return explicit
    name = collection_name or (config or {}).get("collection_name", "obsidian_markdown")
    return os.path.join("./lexical_index", f"{name}.db")


def get_lexical(config, collection_name=None):
    """Return a cached :class:`LexicalIndex` for this profile's lexical db.

    Raises a clear, actionable error if the index has not been built yet — the
    hybrid path needs a BM25 index over the same chunks as the vector store.
    """
    path = lexical_path_for(config, collection_name)
    cached = _LEXICAL_CACHE.get(path)
    if cached is None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Lexical index not found at {path!r}. Build it first with "
                f"`make build-lexical` (rag-build-lexical) — hybrid retrieval needs "
                f"a BM25 index over the same chunks as the vector store."
            )
        cached = LexicalIndex(path)
        _LEXICAL_CACHE[path] = cached
    return cached


def main():
    """``rag-build-lexical`` — build the BM25 index from the vector store."""
    parser = argparse.ArgumentParser(
        description="Build the BM25 lexical index (SQLite FTS5) from the existing "
                    "retrieval collection, for client-side hybrid (rag-query --hybrid).",
    )
    parser.add_argument("--collection", default=None,
                        help="Override the collection name from config")
    parser.add_argument("--out", default=None,
                        help="Override the output db path (default: "
                             "./lexical_index/<collection>.db)")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config, console=True)

    store = get_store(config, args.collection)
    count = store.count()
    if count == 0:
        raise RuntimeError(
            "Collection is empty — build the vector index first (rag-index). "
            "The lexical index is built from what is already in the store."
        )

    path = args.out or lexical_path_for(config, args.collection)
    logger.info("Building lexical index from %d chunks -> %s", count, path)
    n = LexicalIndex.build(store.iter_records(), path)
    logger.info("Lexical index built: %d rows at %s", n, path)
    print(f"Lexical index built: {n} rows at {path}")


if __name__ == "__main__":
    main()
