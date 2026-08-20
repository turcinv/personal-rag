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
import tempfile

from .locking import IndexLockedError, IndexWriterLock
from .provenance import IndexState, read_index_state
from .retrieval import RetrievalFilter
from .utils import load_config, setup_logging
from .store import get_store

logger = logging.getLogger("rag")

# One row per Chroma chunk. `document` is the only analyzed column; `chunk_id`
# and `metadata` (a JSON blob) ride along UNINDEXED. porter stems English
# (troubleshoot↔troubleshooting); unicode61 folds diacritics.
_SCHEMA = """
CREATE TABLE rag_metadata(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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

# Reuse one read connection per canonical db path across queries. Each entry also
# carries the file identity so an atomic replacement is reopened automatically.
_LEXICAL_CACHE: dict = {}
_LEXICAL_SCHEMA_VERSION = "1"


def _canonical_path(path):
    return os.path.realpath(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _path_signature(path):
    stat = os.stat(path)
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _evict_lexical(path):
    cached = _LEXICAL_CACHE.pop(_canonical_path(path), None)
    if cached is not None:
        cached[1].close()


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

    Handles exactly the shapes the legacy ``build_where`` emitted: a single
    ``{field: {"$eq": v}}`` clause or ``{"$and": [clauses...]}``. Retained for the
    raw-dict code path; the neutral :class:`~rag.retrieval.RetrievalFilter` path
    uses ``filter.matches_scalar`` instead.
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


def _where_matches(where, meta):
    """Backend-neutral predicate over one candidate's metadata.

    ``where`` is a :class:`~rag.retrieval.RetrievalFilter` (neutral), a legacy
    Chroma where-dict, or ``None`` — the lexical adapter accepts all three so it
    stays in lockstep with whatever ``query.search`` forwards.
    """
    if where is None:
        return True
    if isinstance(where, RetrievalFilter):
        return where.matches_scalar(meta)
    return _matches_where(meta, where)


class LexicalIndexStaleError(RuntimeError):
    """Raised when a lexical sidecar does not match the vector generation."""


class LexicalIndex:
    """Read-only BM25 view over one FTS5 lexical index file."""

    def __init__(self, path):
        self.path = _canonical_path(path)
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        # URI read-only mode prevents a query process from creating sidecars or
        # mutating an immutable publication. check_same_thread=False supports
        # sharing the cached reader between API request threads.
        self._conn = sqlite3.connect(
            f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
        )

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def build_metadata(self):
        try:
            rows = self._conn.execute("SELECT key, value FROM rag_metadata").fetchall()
        except sqlite3.OperationalError:
            return {}
        return dict(rows)

    def require_fresh(self, index_state: IndexState):
        metadata = self.build_metadata()
        generation = metadata.get("vector_generation_id")
        fingerprint = metadata.get("provenance_fingerprint")
        if (
            metadata.get("schema_version") != _LEXICAL_SCHEMA_VERSION
            or metadata.get("complete") != "true"
            or generation != index_state.generation_id
            or fingerprint != index_state.provenance_fingerprint
        ):
            raise LexicalIndexStaleError(
                "Lexical index is stale for the active vector generation. "
                "Rebuild it with `make build-lexical`."
            )
        return self

    @staticmethod
    def build(
        records,
        path,
        batch_size=5000,
        index_state=None,
        expected_count=None,
        state_reader=None,
    ):
        """Build and validate a sibling temporary DB, then atomically publish it."""
        configured_path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if os.path.islink(configured_path):
            raise RuntimeError(
                f"Refusing to replace lexical symlink: {configured_path}"
            )
        path = _canonical_path(configured_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=parent or "."
        )
        os.close(descriptor)
        os.remove(temporary)

        if isinstance(index_state, IndexState):
            generation_id = index_state.generation_id
            fingerprint = index_state.provenance_fingerprint
        elif index_state is not None:
            generation_id = str(index_state["generation_id"])
            fingerprint = str(index_state["provenance_fingerprint"])
        else:
            generation_id = ""
            fingerprint = ""

        try:
            conn = sqlite3.connect(temporary)
            try:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.executescript(_SCHEMA)
                conn.executemany(
                    "INSERT INTO rag_metadata(key, value) VALUES (?, ?)",
                    [
                        ("schema_version", _LEXICAL_SCHEMA_VERSION),
                        ("complete", "false"),
                        ("vector_generation_id", generation_id),
                        ("provenance_fingerprint", fingerprint),
                    ],
                )
                cur = conn.cursor()
                n = 0
                batch = []
                for cid, doc, meta in records:
                    batch.append(
                        (cid, doc or "", json.dumps(meta or {}, ensure_ascii=False))
                    )
                    if len(batch) >= batch_size:
                        cur.executemany(
                            "INSERT INTO chunks(chunk_id, document, metadata) "
                            "VALUES (?, ?, ?)",
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
                if expected_count is not None and n != expected_count:
                    raise RuntimeError(
                        f"Lexical row count mismatch: built {n}, expected {expected_count}"
                    )
                conn.execute(
                    "INSERT INTO rag_metadata(key, value) VALUES (?, ?)",
                    ("chunk_count", str(n)),
                )
                conn.execute(
                    "UPDATE rag_metadata SET value = 'true' WHERE key = 'complete'"
                )
                conn.commit()
            finally:
                conn.close()

            validation = sqlite3.connect(f"file:{temporary}?mode=ro", uri=True)
            try:
                check = validation.execute("PRAGMA quick_check").fetchone()[0]
                if check != "ok":
                    raise RuntimeError(f"Lexical SQLite quick_check failed: {check}")
                rows = validation.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                metadata = dict(
                    validation.execute("SELECT key, value FROM rag_metadata").fetchall()
                )
                if rows != n or metadata.get("chunk_count") != str(n):
                    raise RuntimeError("Lexical index count validation failed")
                if (
                    metadata.get("schema_version") != _LEXICAL_SCHEMA_VERSION
                    or metadata.get("complete") != "true"
                ):
                    raise RuntimeError("Lexical index metadata validation failed")
            finally:
                validation.close()

            if state_reader is not None and index_state is not None:
                current_state = state_reader()
                if current_state != index_state:
                    raise RuntimeError(
                        "Vector index changed while lexical data was being built; retry"
                    )
            os.replace(temporary, path)
            for suffix in ("-wal", "-shm"):
                stale = path + suffix
                if os.path.exists(stale):
                    os.remove(stale)
            _evict_lexical(path)
            return n
        except Exception:
            for suffix in ("", "-wal", "-shm"):
                stale = temporary + suffix
                if os.path.exists(stale):
                    os.remove(stale)
            raise

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
            if not _where_matches(where, meta):
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


def get_lexical(config, collection_name=None, index_state=None):
    """Return a cached reader, reopening it after an atomic DB replacement."""
    configured_path = lexical_path_for(config, collection_name)
    path = _canonical_path(configured_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Lexical index not found at {configured_path!r}. Build it first with "
            f"`make build-lexical` (rag-build-lexical) — hybrid retrieval needs "
            f"a BM25 index over the same chunks as the vector store."
        )
    signature = _path_signature(path)
    entry = _LEXICAL_CACHE.get(path)
    if entry is None or entry[0] != signature:
        if entry is not None:
            entry[1].close()
        cached = LexicalIndex(path)
        _LEXICAL_CACHE[path] = (signature, cached)
    else:
        cached = entry[1]
    if index_state is None:
        raise LexicalIndexStaleError(
            "Vector generation metadata is required to validate the lexical index"
        )
    return cached.require_fresh(index_state)


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
    parser.add_argument("--wait-for-lock", type=float, default=0.0, metavar="SECONDS",
                        help="Wait up to SECONDS for the index writer lock instead "
                             "of failing fast")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config, console=True)

    index_path = config.get("index_path", "./chroma_db")
    # The lexical build reads the vector store's records + generation and then
    # publishes a generation-bound sidecar; hold the same writer lock as the
    # indexer so a concurrent reindex can't bump the generation mid-build.
    try:
        lock = IndexWriterLock(
            index_path, "rag-build-lexical", timeout=args.wait_for_lock
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(str(exc))
    try:
        lock.acquire()
    except IndexLockedError as exc:
        raise SystemExit(str(exc))
    try:
        store = get_store(config, args.collection)
        state = read_index_state(store)
        if state is None:
            raise RuntimeError(
                "Vector index has no provenance/generation metadata. Rebuild or "
                "explicitly adopt index provenance before building lexical data."
            )
        count = store.count()
        if count == 0:
            raise RuntimeError(
                "Collection is empty — build the vector index first (rag-index). "
                "The lexical index is built from what is already in the store."
            )

        path = args.out or lexical_path_for(config, args.collection)
        logger.info("Building lexical index from %d chunks -> %s", count, path)
        n = LexicalIndex.build(
            store.iter_records(),
            path,
            index_state=state,
            expected_count=count,
            state_reader=lambda: read_index_state(store),
        )
    finally:
        lock.release()
    logger.info("Lexical index built: %d rows at %s", n, path)
    print(f"Lexical index built: {n} rows at {path}")


if __name__ == "__main__":
    main()
