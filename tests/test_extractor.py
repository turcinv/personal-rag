"""Unit tests for the extractor package (extractor.*)."""

import json
import os
import sqlite3
import sys

import pytest

from extractor.analyze_files import detect_type, pdf_has_text, epub_has_text
from extractor.build_index_documents import split_list, to_int, merge
from extractor.build_vault_index import parse_frontmatter, norm_tags
from extractor.dup_detect import shingles, jaccard
from extractor.extract_text import (
    detect_type as ext_detect_type,
    _strip_html,
    _cached_extraction_ok,
    MIN_USABLE_CHARS,
)


# ── extract_text ─────────────────────────────────────────────────────────────

def test_detect_type_md(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# hello")
    assert ext_detect_type(str(p)) == "MD"


def test_detect_type_txt(tmp_path):
    p = tmp_path / "readme.txt"
    p.write_text("plain text")
    assert ext_detect_type(str(p)) == "TXT"


def test_detect_type_unknown(tmp_path):
    p = tmp_path / "binary.bin"
    p.write_bytes(b"\x00\x01\x02")
    assert ext_detect_type(str(p)) == "UNKNOWN"


def test_strip_html_removes_tags():
    result = _strip_html("<p>Hello <b>world</b></p>")
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result


def test_strip_html_removes_script():
    result = _strip_html("<script>alert(1)</script><p>content</p>")
    assert "alert" not in result
    assert "content" in result


# ── extract_text quality gate (_cached_extraction_ok) ─────────────────────────

def _write_cache(tmp_path, name="book.json", **fields):
    p = tmp_path / name
    p.write_text(json.dumps(fields), encoding="utf-8")
    return str(p)


def test_cached_ok_good_cache(tmp_path):
    """Non-empty text well above the threshold, no OCR → good cache (skip)."""
    out = _write_cache(
        tmp_path,
        text="real content " * 100,
        total_chars=1300,
        text_layer_chars=1300,
        ocr_used=False,
        ocr_chars=0,
    )
    assert _cached_extraction_ok(out) is True


def test_cached_ok_empty_text_reextract(tmp_path):
    """Empty text → caught by the no-non-empty-text rule (re-extract)."""
    out = _write_cache(
        tmp_path,
        text="",
        total_chars=0,
        text_layer_chars=0,
        ocr_used=False,
        ocr_chars=0,
    )
    assert _cached_extraction_ok(out) is False


def test_cached_ok_failed_ocr_signature_reextract(tmp_path):
    """OCR attempted but produced nothing and no usable text layer → re-extract.

    A whitespace-only text is non-empty-looking but strips to empty; combined with
    the failed-OCR signature this is exactly the pre-tesseract scanned-PDF case.
    """
    out = _write_cache(
        tmp_path,
        text="   \n  ",
        total_chars=6,
        text_layer_chars=3,
        ocr_used=True,
        ocr_chars=0,
    )
    assert _cached_extraction_ok(out) is False


def test_cached_ok_corrupt_json_reextract(tmp_path):
    """Unreadable / non-JSON file, and a nonexistent path → re-extract."""
    bad = tmp_path / "corrupt.json"
    bad.write_bytes(b"\x00\x01 not json {{{")
    assert _cached_extraction_ok(str(bad)) is False
    assert _cached_extraction_ok(str(tmp_path / "does-not-exist.json")) is False


def test_cached_ok_short_but_valid_is_skipped(tmp_path):
    """A genuinely short but real text-layer document must be SKIPPED, not
    re-extracted — total_chars below MIN_USABLE_CHARS is NOT a standalone trigger."""
    out = _write_cache(
        tmp_path,
        text="A short but perfectly real note.",
        total_chars=120,
        text_layer_chars=120,
        ocr_used=False,
        ocr_chars=0,
    )
    assert 120 < MIN_USABLE_CHARS  # guard: the case is actually below the threshold
    assert _cached_extraction_ok(out) is True


# ── analyze_files ─────────────────────────────────────────────────────────────

def test_analyze_detect_type_pdf(tmp_path):
    p = tmp_path / "book.pdf"
    p.write_bytes(b"%PDF-1.4 fake content")
    assert detect_type(str(p)) == "PDF"


def test_analyze_detect_type_unknown(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    assert detect_type(str(p)) == "UNKNOWN"


# ── build_index_documents ─────────────────────────────────────────────────────

def test_split_list_comma_separated():
    assert split_list("a, b, c") == ["a", "b", "c"]


def test_split_list_semicolon():
    assert split_list("x;y;z") == ["x", "y", "z"]


def test_split_list_empty():
    assert split_list(None) == []
    assert split_list("") == []


def test_to_int_valid():
    assert to_int("42") == 42
    assert to_int(100) == 100


def test_to_int_invalid():
    assert to_int("n/a") is None
    assert to_int(None) is None


def test_merge_builds_record():
    inv = {
        "file_name": "book.pdf", "file_type": "pdf", "source_group": "books",
        "title": "My Book", "author": "A. Author", "primary_topic": "DevOps",
        "tags": "docker, k8s", "confidence": "high", "skill_level": "intermediate",
    }
    ext = {
        "text": "some content here", "total_chars": 17, "total_pages": 5,
        "ocr_used": False, "ocr_pages": [], "ocr_page_count": 0, "ocr_chars": 0,
        "extracted_at": "2026-01-01T00:00:00",
    }
    rec = merge(inv, ext)
    assert rec["title"] == "My Book"
    assert rec["primary_topic"] == "DevOps"
    assert rec["text"] == "some content here"
    assert rec["extraction"]["total_pages"] == 5


# ── build_vault_index ─────────────────────────────────────────────────────────

def test_parse_frontmatter_scalar_fields():
    text = "---\ntitle: My Note\ndomain: DevOps\nconfidence: high\n---\n# Body\nContent here."
    meta, body = parse_frontmatter(text)
    assert meta["title"] == "My Note"
    assert meta["domain"] == "DevOps"
    assert meta["confidence"] == "high"
    assert "Content here" in body


def test_parse_frontmatter_tag_list():
    text = "---\ntags:\n  - docker\n  - k8s\n---\nBody."
    meta, body = parse_frontmatter(text)
    assert meta["tags"] == ["docker", "k8s"]


def test_parse_frontmatter_no_frontmatter():
    text = "# Just a heading\nsome body"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert "Just a heading" in body


def test_norm_tags_list():
    assert norm_tags(["Docker", "K8s"]) == ["docker", "k8s"]


def test_norm_tags_string():
    tags = norm_tags("docker, kubernetes")
    assert "docker" in tags
    assert "kubernetes" in tags


def test_norm_tags_strips_hash():
    assert norm_tags(["#docker"]) == ["docker"]


# ── dup_detect ────────────────────────────────────────────────────────────────

def test_shingles_produces_set():
    result = shingles("one two three four five six seven")
    assert isinstance(result, set)
    assert len(result) > 0


def test_shingles_too_short():
    assert shingles("one two") == set()


def test_jaccard_identical():
    s = shingles("a b c d e f g h i j k l m")
    assert jaccard(s, s) == pytest.approx(1.0)


def test_jaccard_disjoint():
    a = shingles("a b c d e f g h i j k l m")
    b = shingles("x y z p q r s t u v w aa bb")
    assert jaccard(a, b) == pytest.approx(0.0)


def test_jaccard_empty():
    assert jaccard(set(), {"a"}) == 0.0
    assert jaccard({"a"}, set()) == 0.0


# ── build_sqlite (schema smoke test) ─────────────────────────────────────────

def test_build_sqlite_creates_tables(tmp_path):
    """Verify build_sqlite produces the expected schema without needing real data."""
    db_path = str(tmp_path / "test.db")
    jsonl_path = tmp_path / "docs.jsonl"
    jsonl_path.write_text(
        json.dumps({
            "id": "abc123", "file_name": "test.pdf", "file_type": "pdf",
            "source_group": "books", "source_bucket": None, "gcs_path": None,
            "file_size_bytes": 1000, "title": "Test Book", "author": "Author",
            "language": "en", "isbn": None, "page_count": 100,
            "resource_type": "book", "primary_topic": "DevOps",
            "secondary_topics": [], "skill_level": "intermediate",
            "tags": ["docker"], "confidence": "high",
            "classification_status": "classified", "classified_at": None,
            "extraction": {
                "extracted_at": None, "total_pages": 100, "total_documents": None,
                "text_layer_chars": 5000, "ocr_used": False, "ocr_pages": [],
                "ocr_page_count": 0, "ocr_chars": 0, "total_chars": 5000,
            },
            "text": "This is test content " * 50,
        }) + "\n",
        encoding="utf-8",
    )

    from extractor.build_sqlite import main as sqlite_main
    sys.argv = ["rag-build-sqlite", "--jsonl", str(jsonl_path), "--db", db_path]
    sqlite_main()

    conn = sqlite3.connect(db_path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert "documents" in tables
    assert "tags" in tables
    assert "doc_tags" in tables
