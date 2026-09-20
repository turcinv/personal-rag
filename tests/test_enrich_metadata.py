"""Unit tests for extractor.enrich_metadata (title/author/language/ISBN recovery).

Covers the pure cleaning/validation helpers, the embedded-metadata readers
(EPUB via a hand-built zip, PDF via PyMuPDF), the text-index lookup helpers, and
an end-to-end main() run asserting the "fill only when empty" conservatism, the
changes report, and the missing-source stat.
"""

import json
import os
import sys
import zipfile

import fitz  # PyMuPDF

from extractor.enrich_metadata import (
    clean_author,
    clean_title,
    epub_meta,
    find_isbn,
    find_source,
    get_text,
    load_text_index,
    main,
    normalize_isbn,
    pdf_meta,
)


# ── clean_title ──────────────────────────────────────────────────────────────

def test_clean_title_good_value_collapses_whitespace():
    assert clean_title("  Deep   Learning  ", "somestem") == "Deep Learning"


def test_clean_title_rejects_junk_words():
    for junk in ("untitled", "Title", "unknown", "Microsoft Word", "document"):
        assert clean_title(junk, "stem") is None


def test_clean_title_rejects_too_short_and_numeric():
    assert clean_title("ab", "stem") is None       # < 3 chars
    assert clean_title("12345", "stem") is None     # all digits


def test_clean_title_rejects_filename_like_and_software():
    assert clean_title("mybook.pdf", "stem") is None
    assert clean_title("draft.docx", "stem") is None
    assert clean_title("Microsoft Word - report", "stem") is None


def test_clean_title_rejects_when_equal_to_stem():
    assert clean_title("myfilestem", "myfilestem") is None
    assert clean_title("MyFileStem", "myfilestem") is None  # case-insensitive


def test_clean_title_empty_and_none():
    assert clean_title("", "stem") is None
    assert clean_title(None, "stem") is None


# ── clean_author ─────────────────────────────────────────────────────────────

def test_clean_author_good_value():
    assert clean_author("  Aurélien  Géron ", "Some Title") == "Aurélien Géron"


def test_clean_author_rejects_software_hints():
    for junk in ("Adobe InDesign", "calibre", "LaTeX", "Ghostscript"):
        assert clean_author(junk, "Title") is None


def test_clean_author_rejects_short_numeric_and_title_echo():
    assert clean_author("A", "Title") is None
    assert clean_author("2024", "Title") is None
    assert clean_author("Deep Learning", "deep learning") is None  # equals title


def test_clean_author_empty_and_none():
    assert clean_author("", "T") is None
    assert clean_author(None, "T") is None


# ── ISBN ─────────────────────────────────────────────────────────────────────

def test_normalize_isbn_strips_separators():
    assert normalize_isbn("978-1-4842-7268-8") == "9781484272688"
    assert normalize_isbn("0 13 468599 7") == "0134685997"


def test_find_isbn_prefers_isbn13():
    text = "front matter\nISBN 978-1-4842-7268-8 more\n" + ("x" * 100)
    assert find_isbn(text) == "9781484272688"


def test_find_isbn_falls_back_to_isbn10():
    text = "some book ISBN: 0-13-468599-7 here"
    assert find_isbn(text) == "0134685997"


def test_find_isbn_searches_head_and_tail_only():
    # ISBN buried in the exact middle (beyond head+tail windows) is not found.
    middle = "9781484272688"
    text = ("a" * 9000) + middle + ("b" * 9000)
    assert find_isbn(text) is None


def test_find_isbn_none_on_empty_or_absent():
    assert find_isbn("") is None
    assert find_isbn("no identifier here") is None


# ── epub_meta (hand-built zip) ───────────────────────────────────────────────

def _make_epub(path, title="T", creator="A", language="en", with_container=True):
    opf = (
        '<?xml version="1.0"?>'
        '<package xmlns:dc="http://purl.org/dc/elements/1.1/">'
        "<metadata>"
        f"<dc:title>{title}</dc:title>"
        f"<dc:creator>{creator}</dc:creator>"
        f"<dc:language>{language}</dc:language>"
        "</metadata></package>"
    )
    with zipfile.ZipFile(path, "w") as z:
        if with_container:
            z.writestr(
                "META-INF/container.xml",
                '<?xml version="1.0"?><container><rootfiles>'
                '<rootfile full-path="content.opf"/></rootfiles></container>',
            )
        z.writestr("content.opf", opf)


def test_epub_meta_reads_title_author_language(tmp_path):
    p = tmp_path / "book.epub"
    _make_epub(str(p), title="Clean Code", creator="Robert C. Martin", language="en")
    title, author, lang = epub_meta(str(p))
    assert title == "Clean Code"
    assert author == "Robert C. Martin"
    assert lang == "en"


def test_epub_meta_no_container_returns_none(tmp_path):
    p = tmp_path / "broken.epub"
    _make_epub(str(p), with_container=False)
    assert epub_meta(str(p)) == (None, None, None)


def test_epub_meta_bad_file_returns_none(tmp_path):
    p = tmp_path / "notzip.epub"
    p.write_bytes(b"not a zip file")
    assert epub_meta(str(p)) == (None, None, None)


# ── pdf_meta (via PyMuPDF) ───────────────────────────────────────────────────

def _make_pdf(path, title="", author=""):
    doc = fitz.open()
    doc.new_page()
    doc.set_metadata({"title": title, "author": author})
    doc.save(str(path))
    doc.close()


def test_pdf_meta_reads_embedded_title_author(tmp_path):
    p = tmp_path / "doc.pdf"
    _make_pdf(p, title="The Pragmatic Programmer", author="Hunt & Thomas")
    title, author, lang = pdf_meta(str(p))
    assert title == "The Pragmatic Programmer"
    assert author == "Hunt & Thomas"
    assert lang is None  # pdf_meta never returns a language


def test_pdf_meta_missing_file_returns_none():
    assert pdf_meta("/nonexistent/path/to/file.pdf") == (None, None, None)


# ── text-index helpers ───────────────────────────────────────────────────────

def test_load_text_index_maps_json_and_skips_reports(tmp_path):
    d = tmp_path / "text_output"
    d.mkdir()
    (d / "a.json").write_text("{}")
    (d / "manifest.json").write_text("{}")
    (d / "build_report.json").write_text("{}")
    (d / "notes.txt").write_text("x")
    idx = load_text_index([str(d), "/does/not/exist"])
    assert set(idx) == {"a.json"}


def test_get_text_returns_text_field(tmp_path):
    d = tmp_path / "text_output"
    d.mkdir()
    (d / "book.json").write_text(json.dumps({"text": "hello world"}), encoding="utf-8")
    idx = load_text_index([str(d)])
    assert get_text(idx, "book.pdf") == "hello world"       # keyed by stem
    assert get_text(idx, "missing.pdf") == ""


def test_find_source_first_match_wins(tmp_path):
    d1 = tmp_path / "books"
    d2 = tmp_path / "resources"
    d1.mkdir()
    d2.mkdir()
    (d2 / "x.pdf").write_bytes(b"%PDF-1.4")
    assert find_source([str(d1), str(d2)], "x.pdf") == str(d2 / "x.pdf")
    assert find_source([str(d1), str(d2)], "nope.pdf") is None


# ── main() end-to-end ────────────────────────────────────────────────────────

def _run_main(inventory, source_dirs, text_dirs, out):
    argv = ["enrich_metadata", "--inventory", str(inventory), "--out", str(out)]
    for d in source_dirs:
        argv += ["--source-dir", str(d)]
    for d in text_dirs:
        argv += ["--text-dir", str(d)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old


def _read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def test_main_enriches_epub_and_isbn_and_writes_report(tmp_path):
    books = tmp_path / "Books"
    text = tmp_path / "text_output"
    books.mkdir()
    text.mkdir()

    _make_epub(str(books / "cleancode.epub"), title="Clean Code",
               creator="Robert C. Martin", language="en")
    (text / "cleancode.json").write_text(
        json.dumps({"text": "front\nISBN 978-1-4842-7268-8\ntail"}), encoding="utf-8")

    inv = tmp_path / "inventory.jsonl"
    inv.write_text(json.dumps({
        "file_name": "cleancode.epub", "file_type": "epub",
        "title": "cleancode", "author": "", "language": "", "resource_type": "book",
    }) + "\n", encoding="utf-8")

    out = tmp_path / "enriched.jsonl"
    _run_main(inv, [books], [text], out)

    rec = _read_jsonl(out)[0]
    assert rec["title"] == "Clean Code"        # stem title replaced
    assert rec["author"] == "Robert C. Martin"
    assert rec["language"] == "en"
    assert rec["isbn"] == "9781484272688"
    assert rec["enriched"] is True

    report = json.loads((tmp_path / "enriched_changes.json").read_text())
    assert report["stats"] == {"title": 1, "author": 1, "language": 1,
                               "isbn": 1, "missing_source": 0}
    assert report["changes"][0]["file_name"] == "cleancode.epub"


def test_main_is_conservative_and_counts_missing_source(tmp_path):
    books = tmp_path / "Books"
    books.mkdir()
    # A good existing title/author must NOT be overwritten; source file present.
    _make_epub(str(books / "present.epub"), title="Embedded Title",
               creator="Embedded Author", language="fr")

    inv = tmp_path / "inventory.jsonl"
    inv.write_text(
        json.dumps({"file_name": "present.epub", "file_type": "epub",
                    "title": "My Curated Title", "author": "Curated Author",
                    "language": "cs"}) + "\n"
        + json.dumps({"file_name": "gone.pdf", "file_type": "pdf",
                      "title": "", "author": ""}) + "\n",
        encoding="utf-8")

    out = tmp_path / "enriched.jsonl"
    _run_main(inv, [books], [], out)

    recs = {r["file_name"]: r for r in _read_jsonl(out)}
    present = recs["present.epub"]
    assert present["title"] == "My Curated Title"   # not overwritten
    assert present["author"] == "Curated Author"
    assert present["language"] == "cs"
    assert present["enriched"] is False

    report = json.loads((tmp_path / "enriched_changes.json").read_text())
    assert report["stats"]["missing_source"] == 1   # gone.pdf has no source
    assert report["stats"]["title"] == 0
