"""Unit tests for extractor.dup_detect (ISBN + content near-duplicate report).

`shingles`/`jaccard` are covered in test_extractor.py; this file covers the
readers (load_inventory, load_texts, vault_link) and the main() end-to-end report
build — ISBN grouping, content-pair detection, and the written Markdown.
"""

import json
import sys

from extractor.dup_detect import load_inventory, load_texts, main, vault_link


# ── load_inventory ───────────────────────────────────────────────────────────

def test_load_inventory_keys_by_file_name_and_skips_blanks(tmp_path):
    inv = tmp_path / "inv.jsonl"
    inv.write_text(
        json.dumps({"file_name": "a.pdf", "title": "A"}) + "\n"
        + "\n"
        + json.dumps({"file_name": "b.pdf", "title": "B"}) + "\n",
        encoding="utf-8",
    )
    out = load_inventory(str(inv))
    assert set(out) == {"a.pdf", "b.pdf"}
    assert out["a.pdf"]["title"] == "A"


# ── load_texts ───────────────────────────────────────────────────────────────

def _write_text(dir_path, json_name, filename, text):
    (dir_path / json_name).write_text(
        json.dumps({"filename": filename, "text": text}), encoding="utf-8")


def test_load_texts_keys_by_filename_field_and_skips_reports(tmp_path):
    d = tmp_path / "text_output"
    d.mkdir()
    _write_text(d, "a.json", "a.pdf", "alpha body")
    (d / "manifest.json").write_text("{}")
    (d / "build_report.json").write_text("{}")
    (d / "notes.txt").write_text("x")
    out = load_texts([str(d), "/does/not/exist"])
    assert out == {"a.pdf": "alpha body"}


# ── vault_link ───────────────────────────────────────────────────────────────

def test_vault_link_books_vs_resources_and_title_fallback():
    inv = {
        "a.pdf": {"source_group": "books", "title": "Alpha"},
        "b.pdf": {"source_group": "resources"},
    }
    assert vault_link(inv, "a.pdf") == "[[Books/a.pdf|Alpha]]"
    # No title → filename stem; non-book source_group → Resources folder.
    assert vault_link(inv, "b.pdf") == "[[Resources/b.pdf|b]]"


# ── main() end-to-end ────────────────────────────────────────────────────────

# Distinct vocab per group so only the intended pairs cross the threshold.
_SHARED = ("neural gradient tensor kernel embedding manifold cluster lattice "
           "vector corpus shingle jaccard entropy cosine sigmoid softmax dropout "
           "epoch batch") * 3
_ISBN_TEXT = "quarterly ledger invoice depreciation amortization equity dividend yield"


def _run_main(inventory, text_dir, out, threshold=0.35):
    argv = ["dup_detect", "--inventory", str(inventory), "--text-dir", str(text_dir),
            "--out", str(out), "--threshold", str(threshold)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old


def test_main_reports_isbn_group_and_content_pair(tmp_path, capsys):
    text = tmp_path / "text_output"
    text.mkdir()
    # Content near-duplicates: near-identical shared passage → high Jaccard.
    _write_text(text, "dup1.json", "dup1.pdf", _SHARED + " unique alpha")
    _write_text(text, "dup2.json", "dup2.pdf", _SHARED + " unique beta")
    # ISBN duplicates: same ISBN, different text → forced pair regardless of Jaccard.
    _write_text(text, "isbnA.json", "isbnA.pdf", _ISBN_TEXT + " north")
    _write_text(text, "isbnB.json", "isbnB.pdf", _ISBN_TEXT + " south")
    # Lone doc: unrelated vocab, no ISBN → no pair.
    _write_text(text, "lone.json", "lone.pdf", "sailboat harbor lighthouse tide "
                "seagull driftwood pebble anchor rope mast")

    inv = tmp_path / "inv.jsonl"
    inv.write_text("".join(json.dumps(r) + "\n" for r in [
        {"file_name": "dup1.pdf", "source_group": "books", "title": "Dup One"},
        {"file_name": "dup2.pdf", "source_group": "books", "title": "Dup Two"},
        {"file_name": "isbnA.pdf", "source_group": "books", "title": "Isbn A",
         "isbn": "9780000000001"},
        {"file_name": "isbnB.pdf", "source_group": "books", "title": "Isbn B",
         "isbn": "9780000000001"},
        {"file_name": "lone.pdf", "source_group": "resources", "title": "Lone"},
    ]), encoding="utf-8")

    out = tmp_path / "Content Duplicate Candidates.md"
    _run_main(inv, text, out)

    md = out.read_text(encoding="utf-8")
    # ISBN section present with the shared ISBN and both members.
    assert "## Exact ISBN matches (high confidence)" in md
    assert "ISBN 9780000000001" in md
    assert "[[Books/isbnA.pdf|Isbn A]]" in md and "[[Books/isbnB.pdf|Isbn B]]" in md
    # Two candidate pairs across five documents (content dup + ISBN pair).
    assert "2 candidate pair(s) across 5 documents" in md
    # The content pair renders in the similarity table.
    assert "[[Books/dup1.pdf|Dup One]]" in md and "[[Books/dup2.pdf|Dup Two]]" in md
    # The ISBN pair is flagged in the table.
    assert "✅" in md

    summary = capsys.readouterr().out
    assert "1 ISBN-duplicate group(s); 2 content pair(s)" in summary


def test_main_no_pairs_when_all_unrelated(tmp_path):
    text = tmp_path / "text_output"
    text.mkdir()
    _write_text(text, "a.json", "a.pdf", "sailboat harbor lighthouse tide seagull")
    _write_text(text, "b.json", "b.pdf", "compiler register opcode syscall mutex")

    inv = tmp_path / "inv.jsonl"
    inv.write_text(
        json.dumps({"file_name": "a.pdf", "source_group": "books"}) + "\n"
        + json.dumps({"file_name": "b.pdf", "source_group": "books"}) + "\n",
        encoding="utf-8")

    out = tmp_path / "dups.md"
    _run_main(inv, text, out)

    md = out.read_text(encoding="utf-8")
    assert "_No pairs above threshold._" in md
    assert "0 candidate pair(s) across 2 documents" in md
    assert "## Exact ISBN matches" not in md  # no ISBN dups → section omitted
