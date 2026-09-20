"""Unit tests for extractor.build_obsidian_notes (helpers, loaders, build_note, main)."""

import json
import sys

from extractor.build_obsidian_notes import (
    EXCERPT_CHARS,
    build_note,
    excerpt,
    existing_notes,
    icon,
    load_extractions,
    load_inventory,
    main,
    norm_lp,
    norm_moc,
    safe_stem,
    split_list,
    to_int,
    vault_folder,
    yaml_escape,
)


def _frontmatter(note):
    """Parse the leading YAML-ish frontmatter block into a {key: raw_value} dict."""
    lines = note.splitlines()
    assert lines[0] == "---"
    fm = {}
    for line in lines[1:]:
        if line == "---":
            break
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip()
    return fm


def _full_inventory_record():
    return {
        "file_name": "cleancode.epub",
        "title": "Clean Code",
        "author": "Robert C. Martin",
        "source_group": "books",
        "resource_type": "book",
        "primary_topic": "Software Craftsmanship",
        "secondary_topics": "Refactoring, Testing",
        "tags": "clean-code, oop",
        "skill_level": "intermediate",
        "page_count": "464",
        "isbn": "9780132350884",
        "language": "en",
        "confidence": "high",
        "status": "classified",
    }


def _full_extraction_record():
    return {
        "filename": "cleancode.epub",
        "extraction": {"total_pages": 464},
        "text": "  Some   extracted\n\ntext here.  ",
        "total_chars": 12345,
        "ocr_used": True,
        "ocr_page_count": 3,
    }


# ── split_list ───────────────────────────────────────────────────────────────

def test_split_list_comma_separated():
    assert split_list("a, b, c") == ["a", "b", "c"]


def test_split_list_semicolon_separated():
    assert split_list("x;y;z") == ["x", "y", "z"]


def test_split_list_drops_empty_segments():
    assert split_list("a, ;b,, c") == ["a", "b", "c"]


def test_split_list_empty_and_none():
    assert split_list("") == []
    assert split_list(None) == []


# ── to_int ───────────────────────────────────────────────────────────────────

def test_to_int_valid():
    assert to_int("42") == 42
    assert to_int(7) == 7
    assert to_int("  9 ") == 9


def test_to_int_invalid_returns_none():
    assert to_int("n/a") is None
    assert to_int(None) is None
    assert to_int("3.5") is None


# ── norm_moc / norm_lp ───────────────────────────────────────────────────────

def test_norm_moc_normalises_ampersand_and_slash():
    assert norm_moc("DevOps") == "Topic MOC - DevOps"
    assert norm_moc("AI & ML") == "Topic MOC - AI and ML"
    assert norm_moc("CI/CD") == "Topic MOC - CI-CD"


def test_norm_lp_keeps_ampersand_normalises_slash():
    assert norm_lp("DevOps") == "Learning Path - DevOps"
    assert norm_lp("CI/CD") == "Learning Path - CI-CD"
    assert norm_lp("AI & ML") == "Learning Path - AI & ML"


# ── yaml_escape ──────────────────────────────────────────────────────────────

def test_yaml_escape_wraps_and_escapes_quotes():
    assert yaml_escape("Clean Code") == '"Clean Code"'
    assert yaml_escape('a "b" c') == '"a \\"b\\" c"'
    assert yaml_escape(42) == '"42"'


# ── excerpt ──────────────────────────────────────────────────────────────────

def test_excerpt_collapses_whitespace():
    assert excerpt("hello   world\n\nfoo\tbar") == "hello world foo bar"


def test_excerpt_empty_and_none():
    assert excerpt("") == ""
    assert excerpt(None) == ""


def test_excerpt_short_has_no_ellipsis():
    assert excerpt("a short note") == "a short note"


def test_excerpt_truncates_long_text():
    result = excerpt("x" * (EXCERPT_CHARS + 100))
    assert result.endswith("…")
    assert len(result) == EXCERPT_CHARS + 1
    assert result[:EXCERPT_CHARS] == "x" * EXCERPT_CHARS


# ── vault_folder / icon ──────────────────────────────────────────────────────

def test_vault_folder():
    assert vault_folder("books") == "Books"
    assert vault_folder("resources") == "Resources"
    assert vault_folder("") == "Resources"


def test_icon():
    assert icon("books") == "📖"
    assert icon("resources") == "📄"


# ── safe_stem ────────────────────────────────────────────────────────────────

def test_safe_stem_strips_extension_and_unsafe_chars():
    assert safe_stem("book.pdf") == "book"
    assert safe_stem("no_ext") == "no_ext"
    assert safe_stem("a:b*c?.md") == "a_b_c_"
    assert safe_stem("sub/dir/file.txt") == "sub_dir_file"


# ── load_inventory ───────────────────────────────────────────────────────────

def test_load_inventory_keys_by_file_name_and_skips_blank(tmp_path):
    p = tmp_path / "inv.jsonl"
    p.write_text(
        json.dumps({"file_name": "a.pdf", "title": "A"}) + "\n"
        + "\n"
        + json.dumps({"file_name": "b.epub", "title": "B"}) + "\n",
        encoding="utf-8")
    inv = load_inventory(str(p))
    assert set(inv) == {"a.pdf", "b.epub"}
    assert inv["a.pdf"]["title"] == "A"


# ── load_extractions ─────────────────────────────────────────────────────────

def test_load_extractions_maps_json_and_skips_the_rest(tmp_path):
    d = tmp_path / "text_output"
    d.mkdir()
    (d / "book.json").write_text(
        json.dumps({"filename": "book.pdf", "text": "t"}), encoding="utf-8")
    (d / "manifest.json").write_text("{}", encoding="utf-8")
    (d / "build_report.json").write_text("{}", encoding="utf-8")
    (d / "notes.txt").write_text("x", encoding="utf-8")
    (d / "nofilename.json").write_text(
        json.dumps({"text": "no key"}), encoding="utf-8")
    ext = load_extractions([str(d), str(tmp_path / "missing")])
    assert set(ext) == {"book.pdf"}
    assert ext["book.pdf"]["text"] == "t"


# ── existing_notes ───────────────────────────────────────────────────────────

def test_existing_notes_collects_md_stems(tmp_path):
    d = tmp_path / "Generated"
    d.mkdir()
    (d / "Topic MOC - DevOps.md").write_text("x", encoding="utf-8")
    (d / "Learning Path - DevOps.md").write_text("x", encoding="utf-8")
    (d / "notes.txt").write_text("x", encoding="utf-8")
    assert existing_notes(str(d)) == {"Topic MOC - DevOps", "Learning Path - DevOps"}


def test_existing_notes_missing_dir_is_empty(tmp_path):
    assert existing_notes(str(tmp_path / "nope")) == set()


# ── build_note ───────────────────────────────────────────────────────────────

def test_build_note_full_frontmatter():
    moc_set = {
        "Topic MOC - Software Craftsmanship",
        "Topic MOC - Testing",
        "Learning Path - Software Craftsmanship",
    }
    note = build_note(_full_inventory_record(), _full_extraction_record(),
                      moc_set, "_catalog/indexed")
    fm = _frontmatter(note)
    assert fm["title"] == '"Clean Code"'
    assert fm["author"] == '"Robert C. Martin"'
    assert fm["type"] == "book"
    assert fm["source_file"] == '"Books/cleancode.epub"'
    assert fm["index_json"] == '"_catalog/indexed/cleancode.json"'
    assert fm["primary_topic"] == '"Software Craftsmanship"'
    assert fm["secondary_topics"] == '["Refactoring", "Testing"]'
    assert fm["skill_level"] == "intermediate"
    assert fm["page_count"] == "464"
    assert fm["isbn"] == "9780132350884"
    assert fm["language"] == "en"
    assert fm["confidence"] == "high"
    assert fm["status"] == "classified"
    assert fm["tags"] == "[clean-code, oop, intermediate, resource-note]"
    assert "created" in fm and "updated" in fm


def test_build_note_full_body_links_and_excerpt():
    moc_set = {
        "Topic MOC - Software Craftsmanship",
        "Topic MOC - Testing",
        "Learning Path - Software Craftsmanship",
    }
    note = build_note(_full_inventory_record(), _full_extraction_record(),
                      moc_set, "_catalog/indexed")
    assert "# Clean Code" in note
    assert "**Source:** [[Books/cleancode.epub|📖 Clean Code]]" in note
    assert "[[Topic MOC - Software Craftsmanship]]" in note
    assert "`Refactoring`" in note
    assert "[[Topic MOC - Testing]]" in note
    assert "**Learning path:** [[Learning Path - Software Craftsmanship]]" in note
    assert "**Tags:** #clean-code #oop" in note
    assert "`book` · **intermediate** · 464p" in note
    assert "> [!info] Extraction" in note
    assert "12,345 chars extracted" in note
    assert "OCR on 3 page(s)" in note
    assert "## Excerpt" in note
    assert "Some extracted text here." in note


def test_build_note_minimal_defaults():
    inv = {"file_name": "note.md"}
    ext = {"filename": "note.md", "text": ""}
    note = build_note(inv, ext, set(), "_catalog/indexed")
    fm = _frontmatter(note)
    assert fm["title"] == '"note"'
    assert "author" not in fm
    assert fm["type"] == "resource"
    assert fm["source_file"] == '"Resources/note.md"'
    assert fm["confidence"] == "unknown"
    assert fm["status"] == "classified"
    assert fm["tags"] == "[resource-note]"
    assert "primary_topic" not in fm
    assert "skill_level" not in fm
    assert "page_count" not in fm
    assert "**Source:** [[Resources/note.md|📄 note]]" in note
    assert "**Topics:**" not in note
    assert "**Learning path:**" not in note
    assert "> [!info] Extraction" not in note
    assert "## Excerpt" in note


def test_build_note_emits_archived_status_without_filtering():
    inv = {"file_name": "old.pdf", "title": "Old", "status": "archived",
           "source_group": "resources"}
    note = build_note(inv, {"filename": "old.pdf", "text": "x"}, set(),
                      "_catalog/indexed")
    # Audit: status is written but never used to filter; 'archived' passes through.
    assert _frontmatter(note)["status"] == "archived"


# ── main() end-to-end ────────────────────────────────────────────────────────

def _run_main(monkeypatch, inventory, text_dirs, generated_dir, out, index_dir=None):
    argv = ["build_obsidian_notes", "--inventory", str(inventory),
            "--generated-dir", str(generated_dir), "--out", str(out)]
    for d in text_dirs:
        argv += ["--text-dir", str(d)]
    if index_dir is not None:
        argv += ["--index-dir", str(index_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    main()


def test_main_writes_notes_including_archived_row(tmp_path, monkeypatch):
    text = tmp_path / "text_output"
    text.mkdir()
    (text / "guide.json").write_text(
        json.dumps({"filename": "guide.pdf", "text": "Guide body text.",
                    "total_chars": 16, "ocr_used": False}), encoding="utf-8")
    (text / "old.json").write_text(
        json.dumps({"filename": "old.pdf", "text": "Old body."}), encoding="utf-8")

    inv = tmp_path / "inventory.jsonl"
    inv.write_text(
        json.dumps({"file_name": "guide.pdf", "title": "The Guide",
                    "source_group": "resources", "resource_type": "guide",
                    "primary_topic": "DevOps", "status": "classified"}) + "\n"
        + json.dumps({"file_name": "old.pdf", "title": "Old Doc",
                      "source_group": "resources", "status": "archived"}) + "\n",
        encoding="utf-8")

    out = tmp_path / "Resource Notes"
    _run_main(monkeypatch, inventory=inv, text_dirs=[text],
              generated_dir=tmp_path / "Generated", out=out)

    guide = out / "guide.md"
    old = out / "old.md"
    assert guide.exists()
    assert old.exists()  # archived row still produces a note (not filtered)
    assert _frontmatter(old.read_text(encoding="utf-8"))["status"] == "archived"
    guide_fm = _frontmatter(guide.read_text(encoding="utf-8"))
    assert guide_fm["title"] == '"The Guide"'
    assert guide_fm["type"] == "guide"


def test_main_skips_extraction_without_inventory_match(tmp_path, monkeypatch):
    text = tmp_path / "text_output"
    text.mkdir()
    (text / "orphan.json").write_text(
        json.dumps({"filename": "orphan.pdf", "text": "no match"}), encoding="utf-8")

    inv = tmp_path / "inventory.jsonl"
    inv.write_text(
        json.dumps({"file_name": "other.pdf", "title": "Other"}) + "\n",
        encoding="utf-8")

    out = tmp_path / "out"
    _run_main(monkeypatch, inventory=inv, text_dirs=[text],
              generated_dir=tmp_path / "gen", out=out)

    assert not (out / "orphan.md").exists()
    assert list(out.glob("*.md")) == []
