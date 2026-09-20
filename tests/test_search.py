"""Unit tests for extractor.search (FTS5 full-text search CLI over resources.db).

Builds a small temporary SQLite database whose schema is reused verbatim from
build_sqlite (documents / documents_fts / tags / doc_tags), fills it with a few
synthetic rows, and exercises connect, fts_query, run_search (query hit plus
every filter and limit truncation), and main()'s list / browse / search output
paths via monkeypatched sys.argv and captured stdout. Fully offline: no model
load, no network, and it never touches the real resources.db.
"""

import json
import sqlite3
import sys

import pytest

from extractor.build_sqlite import INDEXES, SCHEMA
from extractor.search import connect, fts_query, main, run_search

# (id, file_name, source_group, title, primary_topic, skill_level, pages, tags, text)
_DOCS = [
    ("d1", "kube.pdf", "books", "Kubernetes in Action", "DevOps",
     "intermediate", 372, ["kubernetes", "docker"],
     "kubernetes ingress controllers route external traffic"),
    ("d2", "threat.pdf", "resources", "Threat Modeling", "Cybersecurity",
     "advanced", 210, ["security"],
     "threat modeling identifies attack surfaces early"),
    ("d3", "pandas.epub", "books", "Pandas Cookbook", "Data",
     "beginner", 421, ["python", "pandas"],
     "pandas dataframe groupby and pivot recipes"),
    ("d4", "docker-notes.md", "vault", "Docker Notes", "DevOps",
     "beginner", None, ["docker"],
     "docker compose spins up multi container stacks"),
    ("d5", "rust.pdf", "resources", "Rust Ownership", "Programming",
     "beginner", 88, ["rust"],
     "rust ownership and borrowing rules explained"),
]


def _make_db(path):
    """Create and populate a small FTS5 catalog DB matching build_sqlite."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
    except sqlite3.OperationalError as exc:  # FTS5 module absent -> fail loudly
        conn.close()
        raise RuntimeError(f"FTS5 unavailable in this sqlite3 build: {exc}") from exc
    conn.executescript(INDEXES)
    tag_ids = {}
    for doc_id, fname, group, title, topic, skill, pages, tags, text in _DOCS:
        conn.execute(
            "INSERT INTO documents (id, file_name, source_group, title, "
            "primary_topic, skill_level, page_count, tags, text) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, fname, group, title, topic, skill, pages, json.dumps(tags), text),
        )
        for tag in tags:
            if tag not in tag_ids:
                conn.execute("INSERT OR IGNORE INTO tags(tag) VALUES (?)", (tag,))
                tag_ids[tag] = conn.execute(
                    "SELECT tag_id FROM tags WHERE tag = ?", (tag,)).fetchone()[0]
            conn.execute(
                "INSERT OR IGNORE INTO doc_tags(doc_id, tag_id) VALUES (?,?)",
                (doc_id, tag_ids[tag]),
            )
    conn.execute(
        "INSERT INTO documents_fts(rowid, title, tags, text) "
        "SELECT rowid, title, tags, text FROM documents"
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "resources.db")
    _make_db(path)
    return path


@pytest.fixture
def conn(db_path):
    connection = connect(db_path)
    yield connection
    connection.close()


def _titles(rows):
    return [r[0] for r in rows]


# --- connect -----------------------------------------------------------------

def test_connect_opens_existing_db(db_path):
    connection = connect(db_path)
    try:
        assert isinstance(connection, sqlite3.Connection)
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 5
    finally:
        connection.close()


def test_connect_missing_db_exits(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        connect(str(tmp_path / "does-not-exist.db"))
    assert "DB not found" in str(excinfo.value)


# --- fts_query ---------------------------------------------------------------

def test_fts_query_ands_bare_terms():
    assert fts_query("kubernetes ingress") == "kubernetes AND ingress"
    assert fts_query("docker") == "docker"
    assert fts_query("  compose   stacks  ") == "compose AND stacks"


def test_fts_query_empty_is_empty():
    assert fts_query("   ") == ""


def test_fts_query_passes_through_operators_and_syntax():
    assert fts_query('"exact phrase"') == '"exact phrase"'
    assert fts_query("docker OR podman") == "docker OR podman"
    assert fts_query("attack NOT defense") == "attack NOT defense"
    assert fts_query("term NEAR other") == "term NEAR other"
    assert fts_query("(a b)") == "(a b)"


def test_fts_query_lowercase_operatorish_words_are_terms():
    # word-boundary + case-sensitive: 'android' and lowercase 'and' are terms.
    assert fts_query("android and sdk") == "android AND and AND sdk"


# --- run_search --------------------------------------------------------------

def test_run_search_query_hit_returns_snippet(conn):
    rows = run_search(conn, "kubernetes ingress", None, None, None, None, 15)
    assert len(rows) == 1
    title, topic, skill, pages, group, fname, hit = rows[0]
    assert title == "Kubernetes in Action"
    assert topic == "DevOps"
    assert skill == "intermediate"
    assert pages == 372
    assert group == "books"
    assert fname == "kube.pdf"
    assert hit and "kubernetes" in hit.lower()


def test_run_search_query_with_source_filter(conn):
    rows = run_search(conn, "docker", None, None, None, "vault", 15)
    assert _titles(rows) == ["Docker Notes"]
    assert rows[0][6] and "docker" in rows[0][6].lower()


def test_run_search_topic_filter_no_query(conn):
    rows = run_search(conn, "", "DevOps", None, None, None, 15)
    assert _titles(rows) == ["Docker Notes", "Kubernetes in Action"]
    assert all(r[6] == "" for r in rows)


def test_run_search_skill_filter(conn):
    rows = run_search(conn, "", None, "beginner", None, None, 15)
    assert _titles(rows) == ["Docker Notes", "Pandas Cookbook", "Rust Ownership"]


def test_run_search_tag_filter(conn):
    assert _titles(run_search(conn, "", None, None, "python", None, 15)) == [
        "Pandas Cookbook"]
    assert _titles(run_search(conn, "", None, None, "docker", None, 15)) == [
        "Docker Notes", "Kubernetes in Action"]


def test_run_search_source_filter(conn):
    assert _titles(run_search(conn, "", None, None, None, "resources", 15)) == [
        "Rust Ownership", "Threat Modeling"]
    assert _titles(run_search(conn, "", None, None, None, "vault", 15)) == [
        "Docker Notes"]


def test_run_search_limit_truncates(conn):
    assert len(run_search(conn, "", None, None, None, None, 15)) == 5
    limited = run_search(conn, "", None, None, None, None, 2)
    assert _titles(limited) == ["Docker Notes", "Kubernetes in Action"]


# --- main(): list / browse / search output ----------------------------------

def _run_main(monkeypatch, db_path, *args):
    monkeypatch.setattr(sys, "argv", ["search", "--db", db_path, *args])
    main()


def test_main_list_topics(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "--list-topics")
    out = capsys.readouterr().out
    assert "   2  DevOps" in out
    for topic in ("Cybersecurity", "Data", "Programming"):
        assert topic in out


def test_main_list_tags(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "--list-tags")
    out = capsys.readouterr().out
    assert "   2  docker" in out
    for tag in ("kubernetes", "python", "pandas", "rust", "security"):
        assert tag in out


def test_main_list_sources(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "--list-sources")
    out = capsys.readouterr().out
    assert "   2  books" in out
    assert "   2  resources" in out
    assert "   1  vault" in out


def test_main_browse_tag_resources_icon(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "--tag", "rust", "--browse")
    out = capsys.readouterr().out
    assert "📄" in out
    assert "Rust Ownership" in out
    assert "rust.pdf" in out
    assert "[Programming · beginner · 88p]" in out
    assert "1 result(s)." in out


def test_main_browse_vault_icon_without_pages(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "--source", "vault", "--browse")
    out = capsys.readouterr().out
    assert "📝" in out
    assert "[DevOps · beginner]" in out  # page_count None -> no 'Np' segment
    assert "docker-notes.md" in out
    assert "1 result(s)." in out


def test_main_query_prints_books_icon_and_hit(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "kubernetes ingress")
    out = capsys.readouterr().out
    assert "📖" in out
    assert "Kubernetes in Action" in out
    assert "kube.pdf" in out
    assert "\x1b[1;33m" in out  # snippet highlight -> hit line was printed
    assert "1 result(s)." in out


def test_main_query_no_matches(monkeypatch, capsys, db_path):
    _run_main(monkeypatch, db_path, "zzqqxx")
    assert "No matches." in capsys.readouterr().out


def test_main_requires_query_or_filter(monkeypatch, capsys, db_path):
    monkeypatch.setattr(sys, "argv", ["search", "--db", db_path])
    with pytest.raises(SystemExit):
        main()
    assert "provide a query" in capsys.readouterr().err
