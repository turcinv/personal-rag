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


SCHEMA = """
PRAGMA journal_mode = WAL;

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, action="append", dest="jsonl_paths",
                    metavar="PATH",
                    help="JSONL input file (repeat for multiple files, e.g. books + vault)")
    ap.add_argument("--db", required=True)
    args = ap.parse_args()

    if os.path.exists(args.db):
        os.remove(args.db)
    for ext in ("-wal", "-shm"):
        if os.path.exists(args.db + ext):
            os.remove(args.db + ext)

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)
    cur = conn.cursor()

    tag_ids = {}
    n = 0
    for jsonl_path in args.jsonl_paths:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                ext = r.get("extraction", {}) or {}
                cur.execute(
                    """INSERT OR REPLACE INTO documents VALUES
                       (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        r.get("id"), r.get("file_name"), r.get("file_type"),
                        r.get("source_group"), r.get("source_bucket"), r.get("gcs_path"),
                        r.get("file_size_bytes"), r.get("title"), r.get("author"),
                        r.get("language"), r.get("isbn"), r.get("page_count"), r.get("resource_type"),
                        r.get("primary_topic"), json.dumps(r.get("secondary_topics", [])),
                        r.get("skill_level"), json.dumps(r.get("tags", [])),
                        r.get("confidence"), r.get("classification_status"),
                        r.get("classified_at"), ext.get("extracted_at"),
                        ext.get("total_pages"), ext.get("total_documents"),
                        ext.get("text_layer_chars"), 1 if ext.get("ocr_used") else 0,
                        ext.get("ocr_page_count"), ext.get("ocr_chars"),
                        ext.get("total_chars"), r.get("text", ""),
                    ),
                )
                doc_id = r.get("id")
                for tag in r.get("tags", []) or []:
                    if tag not in tag_ids:
                        cur.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                        cur.execute("SELECT tag_id FROM tags WHERE tag = ?", (tag,))
                        tag_ids[tag] = cur.fetchone()[0]
                    cur.execute(
                        "INSERT OR IGNORE INTO doc_tags(doc_id, tag_id) VALUES (?,?)",
                        (doc_id, tag_ids[tag]),
                    )
                n += 1

    # Populate the FTS index from the base table.
    cur.execute(
        "INSERT INTO documents_fts(rowid, title, tags, text) "
        "SELECT rowid, title, tags, text FROM documents"
    )
    conn.executescript(INDEXES)
    conn.commit()

    # Quick stats.
    docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    tags = cur.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    topics = cur.execute(
        "SELECT COUNT(DISTINCT primary_topic) FROM documents"
    ).fetchone()[0]
    conn.execute("PRAGMA optimize")
    conn.close()
    print(f"Built {args.db}: {docs} documents, {tags} distinct tags, {topics} primary topics")


if __name__ == "__main__":
    main()
