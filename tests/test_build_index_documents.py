"""End-to-end tests for extractor.build_index_documents.main (join + stem collision)."""

import json
import sys

import pytest

from extractor.build_index_documents import main


def _run_main(inventory, text_dirs, out):
    """Invoke main() with a synthesized argv, restoring sys.argv afterwards."""
    argv = ["build_index_documents", "--inventory", str(inventory), "--out", str(out)]
    for d in text_dirs:
        argv += ["--text-dir", str(d)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old


def _write_inventory(path, rows):
    """Write the resource_inventory.jsonl shape; a trailing blank line is tolerated."""
    lines = [json.dumps(r) for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def _write_extraction(path, filename, text="real content"):
    """Write one extract_text.py-style per-file JSON keyed by an internal filename."""
    rec = {
        "filename": filename,
        "text": text,
        "total_chars": len(text),
        "total_pages": 1,
        "total_documents": 1,
        "text_layer_chars": len(text),
        "ocr_used": False,
        "ocr_pages": [],
        "ocr_page_count": 0,
        "ocr_chars": 0,
        "extracted_at": "2026-01-01T00:00:00",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_main_e2e_writes_perfile_docs_jsonl_and_report(tmp_path):
    text = tmp_path / "text_output"
    text.mkdir()
    _write_extraction(text / "alpha.json", "alpha.pdf", text="alpha body")
    _write_extraction(text / "beta.json", "beta.epub", text="beta body")
    # Ignored: report/manifest by name, a record with no filename, a non-json file.
    (text / "manifest.json").write_text("{}", encoding="utf-8")
    (text / "build_report.json").write_text("{}", encoding="utf-8")
    (text / "nofilename.json").write_text(json.dumps({"text": "x"}), encoding="utf-8")
    (text / "notes.txt").write_text("x", encoding="utf-8")

    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, [
        {"file_name": "alpha.pdf", "file_type": "pdf", "title": "Alpha",
         "author": "A. One", "primary_topic": "DevOps", "tags": "docker, k8s",
         "page_count": "100", "file_size": "2048"},
        {"file_name": "beta.epub", "file_type": "epub", "title": "Beta",
         "author": "B. Two", "primary_topic": "ML"},
    ])

    out = tmp_path / "indexed"
    # A non-existent text-dir must be tolerated by the isdir guard.
    _run_main(inv, [text, tmp_path / "missing_dir"], out)

    alpha = json.loads((out / "alpha.json").read_text(encoding="utf-8"))
    beta = json.loads((out / "beta.json").read_text(encoding="utf-8"))
    assert alpha["title"] == "Alpha"
    assert alpha["tags"] == ["docker", "k8s"]
    assert alpha["text"] == "alpha body"
    assert alpha["page_count"] == 100
    assert alpha["file_size_bytes"] == 2048
    assert alpha["extraction"]["total_pages"] == 1
    assert beta["file_type"] == "epub"

    docs = _read_jsonl(out / "index_documents.jsonl")
    assert len(docs) == 2
    assert {d["file_name"] for d in docs} == {"alpha.pdf", "beta.epub"}

    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert report["matched"] == 2
    assert report["inventory_records"] == 2
    assert report["extraction_records"] == 2
    assert report["in_inventory_without_text"] == []
    assert report["extracted_without_inventory"] == []


def test_main_unmatched_inventory_and_extraction_are_reported(tmp_path):
    text = tmp_path / "text_output"
    text.mkdir()
    _write_extraction(text / "alpha.json", "alpha.pdf")
    _write_extraction(text / "orphan.json", "orphan.pdf")  # extracted, no inventory

    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, [
        {"file_name": "alpha.pdf", "file_type": "pdf", "title": "Alpha"},
        {"file_name": "ghost.pdf", "file_type": "pdf", "title": "Ghost"},  # no text
    ])

    out = tmp_path / "indexed"
    _run_main(inv, [text], out)

    assert (out / "alpha.json").exists()
    assert not (out / "orphan.json").exists()   # no inventory row -> not emitted
    assert not (out / "ghost.json").exists()     # no extracted text -> not emitted

    docs = _read_jsonl(out / "index_documents.jsonl")
    assert [d["file_name"] for d in docs] == ["alpha.pdf"]

    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    assert report["matched"] == 1
    assert report["extracted_without_inventory"] == ["orphan.pdf"]
    assert report["in_inventory_without_text"] == ["ghost.pdf"]


def test_main_stem_collision_is_a_hard_fail_with_no_partial_output(tmp_path):
    text = tmp_path / "text_output"
    text.mkdir()
    # Distinct on-disk names, but the internal filenames share the "book" stem.
    _write_extraction(text / "book_epub.json", "book.epub", text="epub body")
    _write_extraction(text / "book_pdf.json", "book.pdf", text="pdf body")

    inv = tmp_path / "inventory.jsonl"
    _write_inventory(inv, [
        {"file_name": "book.epub", "file_type": "epub", "title": "Book"},
        {"file_name": "book.pdf", "file_type": "pdf", "title": "Book"},
    ])

    out = tmp_path / "indexed"
    with pytest.raises(SystemExit) as exc:
        _run_main(inv, [text], out)

    msg = str(exc.value)
    assert "stem collision" in msg
    assert "book.pdf" in msg and "book.epub" in msg

    # The guard runs BEFORE any write, so no per-file doc and no combined JSONL exist.
    assert list(out.glob("*.json")) == []
    assert not (out / "index_documents.jsonl").exists()
