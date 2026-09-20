"""Unit tests for extractor.build_books_index (Books Index aggregate note builder).

Covers the pure helpers (split_list, to_int), the inventory reader (load_books,
book-only filtering) and the hardcoded 'processed' status in the emitted
frontmatter, the created-date preservation helper, the per-book entry formatter,
the markdown builder (frontmatter, header count, descending-count topic ordering,
summary table), and an end-to-end main() run via a monkeypatched sys.argv that
turns a tiny enriched inventory .jsonl into a Books Index .md.
"""

import json
import re
import sys
import time

import pytest

from extractor.build_books_index import (
    FRONTMATTER_DEFAULTS,
    build_markdown,
    existing_created_date,
    format_entry,
    load_books,
    main,
    split_list,
    to_int,
)


# ── split_list ───────────────────────────────────────────────────────────────

def test_split_list_comma_and_semicolon():
    assert split_list("a, b, c") == ["a", "b", "c"]
    assert split_list("x;y;z") == ["x", "y", "z"]
    assert split_list("a, b; c") == ["a", "b", "c"]


def test_split_list_strips_and_drops_empty_pieces():
    assert split_list("  a ; ; , b  ") == ["a", "b"]


def test_split_list_empty_and_none():
    assert split_list("") == []
    assert split_list(None) == []


# ── to_int ───────────────────────────────────────────────────────────────────

def test_to_int_valid_positive():
    assert to_int("42") == 42
    assert to_int(100) == 100
    assert to_int("  7 ") == 7


def test_to_int_zero_and_negative_are_none():
    assert to_int("0") is None
    assert to_int(-5) is None


def test_to_int_invalid_is_none():
    assert to_int("n/a") is None
    assert to_int(None) is None


# ── load_books ───────────────────────────────────────────────────────────────

def test_load_books_includes_only_book_rows_and_skips_blank_lines(tmp_path):
    inv = tmp_path / "inventory.jsonl"
    inv.write_text(
        json.dumps({"file_name": "a.pdf", "resource_type": "book", "title": "A"}) + "\n"
        + "\n"  # blank line must be skipped, not parsed
        + json.dumps({"file_name": "b.mp4", "resource_type": "video", "title": "B"}) + "\n"
        + json.dumps({"file_name": "c.epub", "resource_type": "book", "title": "C"}) + "\n",
        encoding="utf-8",
    )
    books = load_books(str(inv))
    assert [b["file_name"] for b in books] == ["a.pdf", "c.epub"]
    assert all(b["resource_type"] == "book" for b in books)


def test_status_processed_is_hardcoded_not_derived_from_input(tmp_path):
    # load_books never sets/alters status; the module hardcodes 'processed' in
    # FRONTMATTER_DEFAULTS and emits it verbatim, even if an input row disagrees.
    assert FRONTMATTER_DEFAULTS["status"] == "processed"
    books = [{"file_name": "a.pdf", "title": "A", "primary_topic": "AI",
              "status": "draft"}]
    md = build_markdown(books, str(tmp_path / "missing.md"))
    assert "status: processed" in md
    assert "status: draft" not in md


# ── existing_created_date ────────────────────────────────────────────────────

def test_existing_created_date_missing_file_returns_none(tmp_path):
    assert existing_created_date(str(tmp_path / "does-not-exist.md")) is None


def test_existing_created_date_reads_prior_created(tmp_path):
    p = tmp_path / "Books Index.md"
    p.write_text(
        "---\ntitle: Books Index\ncreated: 2024-01-15\nupdated: 2024-06-01\n---\n",
        encoding="utf-8",
    )
    assert existing_created_date(str(p)) == "2024-01-15"


def test_existing_created_date_no_created_field_returns_none(tmp_path):
    p = tmp_path / "Books Index.md"
    p.write_text("---\ntitle: Books Index\nupdated: 2024-06-01\n---\n", encoding="utf-8")
    assert existing_created_date(str(p)) is None


# ── format_entry ─────────────────────────────────────────────────────────────

def test_format_entry_full_title_skill_pages_and_tags():
    book = {
        "file_name": "clean_code.pdf",
        "title": "Clean Code",
        "skill_level": "intermediate",
        "page_count": 464,
        "tags": "python, craftsmanship",
    }
    assert format_entry(book) == (
        "- **Clean Code** · Intermediate · 464p\n"
        "  [[Books/clean_code.pdf|📖 clean_code]]\n"
        "  *python, craftsmanship*"
    )


def test_format_entry_falls_back_to_stem_unknown_skill_and_no_pages():
    book = {"file_name": "the_book.epub"}
    assert format_entry(book) == (
        "- **the_book** · Unknown · —\n"
        "  [[Books/the_book.epub|📖 the_book]]"
    )


def test_format_entry_whitespace_skill_level_displays_unknown():
    # A whitespace-only skill_level strips to empty → the "Unknown" fallback.
    book = {"file_name": "x.pdf", "title": "X", "skill_level": "   ", "page_count": "3"}
    assert format_entry(book).startswith("- **X** · Unknown · 3p")


# ── build_markdown ───────────────────────────────────────────────────────────

def _book(file_name, title, topic=None, **extra):
    book = {"file_name": file_name, "title": title}
    if topic is not None:
        book["primary_topic"] = topic
    book.update(extra)
    return book


def test_build_markdown_header_count_and_descending_topic_order(tmp_path):
    books = [
        _book("ml1.pdf", "ML One", "Machine Learning"),
        _book("ml2.pdf", "ML Two", "Machine Learning"),
        _book("ml3.pdf", "ML Three", "Machine Learning"),
        _book("ai1.pdf", "AI One", "AI"),
        _book("dev1.pdf", "Dev One", "DevOps"),
    ]
    md = build_markdown(books, str(tmp_path / "missing.md"))

    # Exactly one H1; H2s are Summary + one per topic, ordered by descending
    # count then ascending topic name (AI and DevOps tie at 1 → AI first).
    assert re.findall(r"^# .+$", md, re.MULTILINE) == ["# Books Index"]
    assert re.findall(r"^## .+$", md, re.MULTILINE) == [
        "## Summary", "## Machine Learning", "## AI", "## DevOps",
    ]

    # Summary table rows carry per-topic counts in the same order.
    rows = re.findall(r"^\| (Machine Learning|AI|DevOps)\s+\| \d+\s+\|$", md, re.MULTILINE)
    assert rows == ["Machine Learning", "AI", "DevOps"]
    assert re.search(r"^\| Machine Learning\s+\| 3\s+\|$", md, re.MULTILINE)
    assert re.search(r"^\| AI\s+\| 1\s+\|$", md, re.MULTILINE)


def test_build_markdown_summary_line_and_frontmatter(tmp_path):
    md = build_markdown([_book("a.pdf", "A", "AI"), _book("b.pdf", "B", "AI")],
                        str(tmp_path / "missing.md"))
    assert md.startswith("---\ntitle: Books Index\n")
    assert "> 2 books catalogued across 1 topics." in md
    assert "domain: Personal Knowledge" in md
    assert "type: Reference" in md


def test_build_markdown_preserves_prior_created_date(tmp_path):
    out = tmp_path / "Books Index.md"
    out.write_text("---\ncreated: 2020-02-02\nupdated: 2020-02-02\n---\n", encoding="utf-8")
    md = build_markdown([_book("a.pdf", "A", "AI")], str(out))
    assert "created: 2020-02-02" in md
    assert f"updated: {time.strftime('%Y-%m-%d')}" in md


def test_build_markdown_created_falls_back_to_today_without_prior_file(tmp_path):
    md = build_markdown([_book("a.pdf", "A", "AI")], str(tmp_path / "absent.md"))
    today = time.strftime("%Y-%m-%d")
    assert f"created: {today}" in md
    assert f"updated: {today}" in md


def test_build_markdown_defaults_missing_topic_to_other(tmp_path):
    md = build_markdown([_book("a.pdf", "A")], str(tmp_path / "missing.md"))
    assert "## Other" in md
    assert "> 1 books catalogued across 1 topics." in md


# ── main() end-to-end ────────────────────────────────────────────────────────

def _write_inventory(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_main_writes_index_and_excludes_non_book_rows(tmp_path, monkeypatch):
    inv = tmp_path / "resource_inventory_enriched.jsonl"
    _write_inventory(inv, [
        {"file_name": "a.pdf", "title": "Alpha", "primary_topic": "AI",
         "resource_type": "book"},
        {"file_name": "b.pdf", "title": "Beta", "primary_topic": "DevOps",
         "resource_type": "book"},
        {"file_name": "c.mp4", "title": "Gamma Video", "primary_topic": "AI",
         "resource_type": "video"},
    ])
    out = tmp_path / "Generated" / "Books Index.md"
    monkeypatch.setattr(
        sys, "argv",
        ["rag-build-books-index", "--inventory", str(inv), "--out", str(out)],
    )

    main()

    md = out.read_text(encoding="utf-8")
    assert re.findall(r"^# .+$", md, re.MULTILINE) == ["# Books Index"]
    assert re.findall(r"^## .+$", md, re.MULTILINE) == [
        "## Summary", "## AI", "## DevOps",
    ]
    assert "> 2 books catalogued across 2 topics." in md
    # The non-book row is excluded entirely (no title, no wikilink).
    assert "Gamma Video" not in md
    assert "Books/c.mp4" not in md


def test_main_raises_systemexit_when_no_book_rows(tmp_path, monkeypatch):
    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, [
        {"file_name": "c.mp4", "title": "Only Video", "resource_type": "video"},
    ])
    out = tmp_path / "Books Index.md"
    monkeypatch.setattr(
        sys, "argv",
        ["rag-build-books-index", "--inventory", str(inv), "--out", str(out)],
    )

    with pytest.raises(SystemExit):
        main()
    assert not out.exists()
