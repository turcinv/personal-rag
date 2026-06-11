#!/usr/bin/env python3
"""Full-text search over the resources.db (FTS5) index.

Examples:
  python3 search.py "kubernetes ingress"
  python3 search.py "threat modeling" --topic Cybersecurity --skill intermediate
  python3 search.py "pandas dataframe" --tag python --limit 5
  python3 search.py "docker compose" --source books
  python3 search.py "CI/CD pipeline" --source vault
  python3 search.py --list-topics
  python3 search.py --list-sources
  python3 search.py --tag rust --browse           # filter only, no FTS query

Default DB path is ~/Documents/knowledge-base-index/resources.db (override with --db).
Sources: books, resources, vault  (use --source to filter by source_group)
"""
import argparse
import os
import re
import sqlite3
import sys

DEFAULT_DB = os.path.expanduser("~/Documents/knowledge-base-index/resources.db")


def connect(db):
    if not os.path.exists(db):
        sys.exit(f"DB not found: {db}\nBuild it with build_sqlite.py or pass --db.")
    return sqlite3.connect(db)


def fts_query(terms):
    """Turn a plain query into an FTS5 MATCH expression (AND of bare terms;
    keep quoted phrases and explicit AND/OR/NOT operators as-is)."""
    if re.search(r'["()]|\b(AND|OR|NOT|NEAR)\b', terms):
        return terms
    words = [w for w in re.split(r"\s+", terms.strip()) if w]
    return " AND ".join(words)


def run_search(conn, query, topic, skill, tag, source, limit):
    where, params = [], []
    if query:
        where.append("documents_fts MATCH ?")
        params.append(fts_query(query))
    if topic:
        where.append("d.primary_topic = ?")
        params.append(topic)
    if skill:
        where.append("d.skill_level = ?")
        params.append(skill)
    if source:
        where.append("d.source_group = ?")
        params.append(source)
    if tag:
        where.append("EXISTS (SELECT 1 FROM doc_tags dt JOIN tags t ON t.tag_id=dt.tag_id "
                     "WHERE dt.doc_id=d.id AND t.tag=?)")
        params.append(tag)

    order = "ORDER BY rank" if query else "ORDER BY d.title"
    snippet = ("snippet(documents_fts, -1, '\033[1;33m', '\033[0m', ' … ', 12)"
               if query else "''")
    sql = f"""
        SELECT d.title, d.primary_topic, d.skill_level, d.page_count,
               d.source_group, d.file_name, {snippet} AS hit
        FROM documents d
        {"JOIN documents_fts f ON f.rowid = d.rowid" if query else ""}
        {"WHERE " + " AND ".join(where) if where else ""}
        {order} LIMIT ?
    """
    params.append(limit)
    return conn.execute(sql, params).fetchall()


def main():
    ap = argparse.ArgumentParser(description="Full-text search over resources.db")
    ap.add_argument("query", nargs="?", default="", help="search terms (FTS5)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--topic", help="filter by primary_topic")
    ap.add_argument("--skill", help="filter by skill_level (beginner/intermediate/advanced)")
    ap.add_argument("--tag", help="filter by an exact tag")
    ap.add_argument("--source", help="filter by source_group (books / resources / vault)")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--browse", action="store_true", help="filter-only listing (no query)")
    ap.add_argument("--list-topics", action="store_true")
    ap.add_argument("--list-tags", action="store_true")
    ap.add_argument("--list-sources", action="store_true",
                    help="show all source_group values and counts")
    args = ap.parse_args()

    conn = connect(args.db)

    if args.list_topics:
        rows = conn.execute(
            "SELECT primary_topic, COUNT(*) c FROM documents "
            "GROUP BY primary_topic ORDER BY c DESC").fetchall()
        for t, c in rows:
            print(f"{c:4}  {t}")
        return
    if args.list_tags:
        rows = conn.execute(
            "SELECT t.tag, COUNT(*) c FROM tags t JOIN doc_tags dt ON dt.tag_id=t.tag_id "
            "GROUP BY t.tag ORDER BY c DESC LIMIT 60").fetchall()
        for t, c in rows:
            print(f"{c:4}  {t}")
        return
    if args.list_sources:
        rows = conn.execute(
            "SELECT source_group, COUNT(*) c FROM documents "
            "GROUP BY source_group ORDER BY c DESC").fetchall()
        for s, c in rows:
            print(f"{c:4}  {s}")
        return

    if not args.query and not (args.browse or args.topic or args.skill or args.tag or args.source):
        ap.error("provide a query, or use --browse with a filter, "
                 "or --list-topics / --list-tags / --list-sources")

    rows = run_search(conn, args.query, args.topic, args.skill, args.tag,
                      args.source, args.limit)
    if not rows:
        print("No matches.")
        return
    for title, topic, skill, pages, group, fname, hit in rows:
        if group == "books":
            icon = "📖"
        elif group == "vault":
            icon = "📝"
        else:
            icon = "📄"
        meta = " · ".join(x for x in [topic, skill, f"{pages}p" if pages else None] if x)
        print(f"{icon} \033[1m{title}\033[0m  [{meta}]")
        if hit:
            print(f"    {hit.strip()}")
        print(f"    \033[2m{fname}\033[0m")
    print(f"\n{len(rows)} result(s).")


if __name__ == "__main__":
    main()
