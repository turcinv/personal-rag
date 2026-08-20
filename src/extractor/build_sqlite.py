#!/usr/bin/env python3
"""Build a SQLite database (with FTS5 full-text search) from the merged index
documents produced by build_index_documents.py and/or build_vault_index.py.

Inputs : one or more JSONL files (pass --jsonl multiple times)
Output: a .db with:
  - documents      : one row per resource, all metadata + full text
  - documents_fts  : FTS5 over title + tags + text (ranked search + snippets)
  - tags, doc_tags : normalized tags for clean filtering

Run (books only):
  python3 build_sqlite.py \
      --jsonl ".../_catalog/indexed/index_documents.jsonl" \
      --db    ".../_catalog/resources.db"

Run (books + vault notes):
  python3 build_sqlite.py \
      --jsonl ".../_catalog/indexed/index_documents.jsonl" \
      --jsonl ".../_catalog/indexed/vault_documents.jsonl" \
      --db    ".../_catalog/resources.db"
"""
import argparse
import json
import os
import sqlite3
import tempfile


SCHEMA = """
PRAGMA journal_mode = DELETE;

DROP TABLE IF EXISTS documents;
CREATE TABLE documents (
    id                TEXT PRIMARY KEY,
    file_name         TEXT,
    file_type         TEXT,
    source_group      TEXT,
    source_bucket     TEXT,
    gcs_path          TEXT,
    file_size_bytes   INTEGER,
    title             TEXT,
    author            TEXT,
    language          TEXT,
    isbn              TEXT,
    page_count        INTEGER,
    resource_type     TEXT,
    primary_topic     TEXT,
    secondary_topics  TEXT,   -- JSON array
    skill_level       TEXT,
    tags              TEXT,    -- JSON array
    confidence        TEXT,
    classification_status TEXT,
    classified_at     TEXT,
    extracted_at      TEXT,
    total_pages       INTEGER,
    total_documents   INTEGER,
    text_layer_chars  INTEGER,
    ocr_used          INTEGER,
    ocr_page_count    INTEGER,
    ocr_chars         INTEGER,
    total_chars       INTEGER,
    text              TEXT
);

DROP TABLE IF EXISTS tags;
CREATE TABLE tags (
    tag_id INTEGER PRIMARY KEY,
    tag    TEXT UNIQUE
);

DROP TABLE IF EXISTS doc_tags;
CREATE TABLE doc_tags (
    doc_id TEXT,
    tag_id INTEGER,
    PRIMARY KEY (doc_id, tag_id)
);

DROP TABLE IF EXISTS documents_fts;
CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, tags, text,
    content='documents',
    content_rowid='rowid',
    tokenize='porter unicode61'
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_docs_primary_topic ON documents(primary_topic);
CREATE INDEX IF NOT EXISTS idx_docs_skill_level   ON documents(skill_level);
CREATE INDEX IF NOT EXISTS idx_docs_source_group  ON documents(source_group);
CREATE INDEX IF NOT EXISTS idx_doc_tags_tag       ON doc_tags(tag_id);
"""


def _validate_database(path):
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite quick_check failed: {check}")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        required = {"documents", "documents_fts", "tags", "doc_tags"}
        if not required <= tables:
            raise RuntimeError(
                f"SQLite database missing tables: {sorted(required - tables)}"
            )
        documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        fts_documents = connection.execute(
            "SELECT COUNT(*) FROM documents_fts"
        ).fetchone()[0]
        if documents != fts_documents:
            raise RuntimeError(
                f"SQLite document/FTS count mismatch: {documents} != {fts_documents}"
            )
        tags = connection.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
        topics = connection.execute(
            "SELECT COUNT(DISTINCT primary_topic) FROM documents"
        ).fetchone()[0]
        return int(documents), int(tags), int(topics)
    finally:
        connection.close()


def _remove_sqlite_files(path):
    for suffix in ("", "-wal", "-shm"):
        candidate = path + suffix
        if os.path.exists(candidate):
            os.remove(candidate)


def build_database(jsonl_paths, destination):
    """Build, validate, and atomically replace one whole-document FTS database."""
    destination = os.path.abspath(os.fspath(destination))
    if os.path.islink(destination):
        raise RuntimeError(
            f"Refusing to replace SQLite compatibility symlink: {destination}"
        )
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=parent
    )
    os.close(descriptor)
    os.remove(temporary)

    try:
        conn = sqlite3.connect(temporary)
        try:
            conn.executescript(SCHEMA)
            cur = conn.cursor()
            tag_ids = {}
            for jsonl_path in jsonl_paths:
                with open(jsonl_path, encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        extraction = record.get("extraction", {}) or {}
                        cur.execute(
                            """INSERT OR REPLACE INTO documents VALUES
                               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (
                                record.get("id"), record.get("file_name"), record.get("file_type"),
                                record.get("source_group"), record.get("source_bucket"), record.get("gcs_path"),
                                record.get("file_size_bytes"), record.get("title"), record.get("author"),
                                record.get("language"), record.get("isbn"), record.get("page_count"), record.get("resource_type"),
                                record.get("primary_topic"), json.dumps(record.get("secondary_topics", [])),
                                record.get("skill_level"), json.dumps(record.get("tags", [])),
                                record.get("confidence"), record.get("classification_status"),
                                record.get("classified_at"), extraction.get("extracted_at"),
                                extraction.get("total_pages"), extraction.get("total_documents"),
                                extraction.get("text_layer_chars"), 1 if extraction.get("ocr_used") else 0,
                                extraction.get("ocr_page_count"), extraction.get("ocr_chars"),
                                extraction.get("total_chars"), record.get("text", ""),
                            ),
                        )
                        document_id = record.get("id")
                        for tag in record.get("tags", []) or []:
                            if tag not in tag_ids:
                                cur.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                                cur.execute("SELECT tag_id FROM tags WHERE tag = ?", (tag,))
                                tag_ids[tag] = cur.fetchone()[0]
                            cur.execute(
                                "INSERT OR IGNORE INTO doc_tags(doc_id, tag_id) VALUES (?,?)",
                                (document_id, tag_ids[tag]),
                            )

            cur.execute(
                "INSERT INTO documents_fts(rowid, title, tags, text) "
                "SELECT rowid, title, tags, text FROM documents"
            )
            conn.executescript(INDEXES)
            conn.execute("PRAGMA optimize")
            conn.commit()
        finally:
            conn.close()

        documents, tags, topics = _validate_database(temporary)
        os.replace(temporary, destination)
        for suffix in ("-wal", "-shm"):
            stale = destination + suffix
            if os.path.exists(stale):
                os.remove(stale)
        return documents, tags, topics
    except Exception:
        _remove_sqlite_files(temporary)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, action="append", dest="jsonl_paths",
                    metavar="PATH",
                    help="JSONL input file (repeat for multiple files, e.g. books + vault)")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    documents, tags, topics = build_database(args.jsonl_paths, args.db)
    print(
        f"Built {args.db}: {documents} documents, {tags} distinct tags, "
        f"{topics} primary topics"
    )


if __name__ == "__main__":
    main()
