"""Unit tests for rag.extractors.pdf.extract_pdf_file.

clean_pdf_title is covered in test_extractors.py; this file covers the extraction
path: page→block accumulation, char-window chunking, heading ranges, metadata
shape, embedded-title vs filename fallback, short-page skipping, and the
read-error branch. PDFs with real extractable text are built with PyMuPDF.
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

from rag.extractors.pdf import extract_pdf_file

_LINE = "the quick brown fox jumps over the lazy dogs"  # ~44 chars


def _make_pdf(path, pages, title=None):
    """pages: list of pages, each a list of text lines."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line)
            y += 14
    if title is not None:
        doc.set_metadata({"title": title})
    doc.save(str(path))
    doc.close()


def test_extract_pdf_happy_path_chunks_and_metadata(tmp_path):
    pdf = tmp_path / "some_book_v2.pdf"
    _make_pdf(pdf, [[_LINE] * 15, [_LINE] * 15])  # two ~660-char pages
    ids, docs, metas, err = extract_pdf_file(pdf, "book", max_chars=500, overlap=50)

    assert err is None
    assert len(ids) == len(docs) == len(metas) >= 2
    assert all(len(i) == 64 for i in ids)          # sha256 hex chunk ids
    assert len(set(ids)) == len(ids)                # unique
    m = metas[0]
    assert m["path"] == "some_book_v2.pdf"
    assert m["source"] == "pdf"
    assert m["type"] == "book"
    assert re.fullmatch(r"p\.\d+(-\d+)?", m["heading"])
    assert docs[0].strip()


def test_extract_pdf_uses_embedded_title(tmp_path):
    pdf = tmp_path / "whatever.pdf"
    _make_pdf(pdf, [[_LINE] * 15], title="The Real Title")
    _, _, metas, err = extract_pdf_file(pdf, "book", max_chars=500, overlap=50)
    assert err is None
    assert metas and all(m["title"] == "The Real Title" for m in metas)


def test_extract_pdf_falls_back_to_filename_title(tmp_path):
    pdf = tmp_path / "some_book_v2.pdf"
    _make_pdf(pdf, [[_LINE] * 15])  # no embedded title
    _, _, metas, err = extract_pdf_file(pdf, "book", max_chars=500, overlap=50)
    assert err is None
    # clean_pdf_title strips the _v2 suffix and title-cases.
    assert metas and metas[0]["title"] == "Some Book"


def test_extract_pdf_block_spanning_two_pages_has_range_heading(tmp_path):
    pdf = tmp_path / "doc.pdf"
    # Two ~280-char pages: neither alone reaches max_chars=500, together they do,
    # so the flushed block spans pages 1-2.
    _make_pdf(pdf, [[_LINE] * 6, [_LINE] * 6])
    _, _, metas, err = extract_pdf_file(pdf, "resource", max_chars=500, overlap=50)
    assert err is None
    assert any(m["heading"] == "p.1-2" for m in metas)


def test_extract_pdf_skips_short_pages(tmp_path):
    pdf = tmp_path / "tiny.pdf"
    _make_pdf(pdf, [["hi"], ["yo"]])  # every page < 40 chars → skipped
    ids, docs, metas, err = extract_pdf_file(pdf, "book", max_chars=500, overlap=50)
    assert err is None
    assert ids == [] and docs == [] and metas == []


def test_extract_pdf_read_error_is_reported_not_raised(tmp_path):
    bogus = tmp_path / "not_really.pdf"
    bogus.write_bytes(b"this is not a pdf at all")
    ids, docs, metas, err = extract_pdf_file(bogus, "book", max_chars=500, overlap=50)
    assert (ids, docs, metas) == ([], [], [])
    assert err is not None and "skip pdf read error" in err
