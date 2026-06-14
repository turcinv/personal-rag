"""Unit tests for the per-source extractors (rag.extractors.*)."""

import json

from rag.extractors.json_doc import extract_json_doc
from rag.extractors.markdown import extract_md_file, should_exclude
from rag.extractors.pdf import clean_pdf_title


# --- Markdown ---------------------------------------------------------------

def test_extract_md_maps_frontmatter_to_metadata(tmp_path):
    note = tmp_path / "n.md"
    note.write_text(
        "---\ntitle: My Note\ndomain: DevOps\ntype: Knowledge\n"
        "confidence: high\nstatus: processed\nsource: ChatGPT\n"
        "tags:\n  - k8s\n  - docker\n---\n# Summary\nContent about [[K3s]].",
        encoding="utf-8",
    )
    ids, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None and docs
    m = metas[0]
    assert m["title"] == "My Note"
    assert m["domain"] == "DevOps"
    assert m["type"] == "Knowledge"
    assert m["confidence"] == "high"
    assert m["status"] == "processed"
    assert m["source"] == "ChatGPT"
    assert m["tags"] == "k8s, docker"
    assert m["heading"] == "Summary"
    assert "K3s" in m["wikilinks"]


def test_extract_md_derives_subdomain_from_subfolder(tmp_path):
    note = tmp_path / "Knowledge" / "Software Engineering" / "Python & Backend Development" / "n.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: N\ndomain: Software Engineering\n---\n# Summary\nContent.", encoding="utf-8")
    _, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None and docs
    assert metas[0]["subdomain"] == "Python & Backend Development"


def test_extract_md_root_note_has_empty_subdomain(tmp_path):
    note = tmp_path / "Knowledge" / "Software Engineering" / "n.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: N\ndomain: Software Engineering\n---\n# Summary\nContent.", encoding="utf-8")
    _, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None and docs
    assert metas[0]["subdomain"] == ""


def test_extract_md_frontmatter_subdomain_used_when_flat(tmp_path):
    note = tmp_path / "Knowledge" / "DevOps" / "n.md"
    note.parent.mkdir(parents=True)
    note.write_text("---\ntitle: N\ndomain: DevOps\nsubdomain: Monitoring & Observability\n---\n# Summary\nContent.", encoding="utf-8")
    _, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None and docs
    assert metas[0]["subdomain"] == "Monitoring & Observability"


def test_extract_md_empty_body_yields_nothing(tmp_path):
    note = tmp_path / "e.md"
    note.write_text("---\ntitle: E\n---\n", encoding="utf-8")
    ids, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None
    assert docs == []


def test_should_exclude_dirs_and_files(tmp_path):
    cfg = {"exclude_dirs": ["Templates"], "exclude_files": ["Home.md"]}
    (tmp_path / "Templates").mkdir()
    assert should_exclude(tmp_path / "Templates" / "t.md", tmp_path, cfg)
    assert should_exclude(tmp_path / "Home.md", tmp_path, cfg)
    assert not should_exclude(tmp_path / "Knowledge" / "k.md", tmp_path, cfg)


def test_should_exclude_filename_patterns_catch_mocs(tmp_path):
    cfg = {"exclude_filename_patterns": ["* MOC.md"]}
    assert should_exclude(tmp_path / "Knowledge" / "DevOps MOC.md", tmp_path, cfg)
    assert should_exclude(tmp_path / "Knowledge" / "Edge & IoT MOC.md", tmp_path, cfg)
    assert not should_exclude(tmp_path / "Knowledge" / "MOC Design Note.md", tmp_path, cfg)


def test_extract_md_strips_nav_tail_but_keeps_links_in_metadata(tmp_path):
    note = tmp_path / "n.md"
    note.write_text(
        "---\ntitle: N\n---\n# Summary\nuses [[K3s|k3s cluster]] daily\n"
        "# Related Topics\n- [[Docker]]\n## Potential New Notes\n- [[Future Note]]",
        encoding="utf-8",
    )
    ids, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None and len(docs) == 1
    # embedded text: tail gone, wikilink syntax resolved to alias
    assert docs[0] == "uses k3s cluster daily"
    # graph signal: links from the FULL body survive in metadata
    assert "Docker" in metas[0]["wikilinks"]
    assert "K3s" in metas[0]["wikilinks"]


def test_extract_md_nav_only_note_yields_nothing(tmp_path):
    note = tmp_path / "nav.md"
    note.write_text(
        "---\ntitle: NavOnly\n---\n# Related Topics\n- [[A]]\n- [[B]]",
        encoding="utf-8",
    )
    ids, docs, metas, err = extract_md_file(note, tmp_path, {}, 1200, 150)
    assert err is None
    assert docs == []


# --- JSON -------------------------------------------------------------------

def test_extract_json_maps_metadata(tmp_path):
    p = tmp_path / "book.json"
    p.write_text(json.dumps({
        "file_name": "book.pdf", "title": "A Book", "resource_type": "book",
        "primary_topic": "Cybersecurity", "tags": ["a", "b"],
        "confidence": "high", "text": "x" * 3000,
    }), encoding="utf-8")
    ids, docs, metas, err = extract_json_doc(p, 1200, 150)
    assert err is None and len(docs) >= 2
    m = metas[0]
    assert m["path"] == "book.pdf"
    assert m["title"] == "A Book"
    assert m["type"] == "book"
    assert m["domain"] == "Cybersecurity"
    assert m["source"] == "pdf"
    assert m["confidence"] == "high"
    assert m["tags"] == "a, b"
    assert m["heading"] == "part 1"


def test_extract_json_short_text_skipped(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"file_name": "s.pdf", "text": "tiny"}), encoding="utf-8")
    ids, docs, metas, err = extract_json_doc(p, 1200, 150)
    assert err is None
    assert docs == []


def test_extract_json_bad_json_returns_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    ids, docs, metas, err = extract_json_doc(p, 1200, 150)
    assert err is not None and "json read error" in err
    assert docs == []


# --- PDF helpers ------------------------------------------------------------

def test_clean_pdf_title_strips_version_and_titlecases():
    assert clean_pdf_title("my_book_v2.pdf") == "My Book"
    assert clean_pdf_title("terraform_upandrunning_3rdedition.pdf") == "Terraform Upandrunning"
