"""Unit tests for extractor.link_mocs (managed backlink-block injection into MOCs).

Covers the string helpers (split_list / norm_moc / norm_lp / safe_stem),
load_inventory, build_block rendering, the idempotent inject (replace between
markers while preserving surrounding curated content), and an end-to-end main()
run driven via sys.argv against a tiny inventory and a few fake MOC / LP files.
"""

import json
import sys

from extractor.link_mocs import (
    BEGIN,
    END,
    build_block,
    inject,
    load_inventory,
    main,
    norm_lp,
    norm_moc,
    safe_stem,
    split_list,
)


def _seed(path, body):
    path.write_text(body, encoding="utf-8")


def _read(path):
    return path.read_text(encoding="utf-8")


def _run_main(inventory, generated_dir, notes_subdir="Resource Notes"):
    argv = ["link_mocs", "--inventory", str(inventory),
            "--generated-dir", str(generated_dir), "--notes-subdir", notes_subdir]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old


# ── split_list ───────────────────────────────────────────────────────────────

def test_split_list_comma_separated():
    assert split_list("a, b, c") == ["a", "b", "c"]


def test_split_list_semicolon_separated():
    assert split_list("x;y;z") == ["x", "y", "z"]


def test_split_list_mixed_delimiters_and_whitespace():
    assert split_list("a; b ,  c") == ["a", "b", "c"]


def test_split_list_drops_empty_segments():
    assert split_list("a,,b;") == ["a", "b"]


def test_split_list_empty_and_none():
    assert split_list("") == []
    assert split_list(None) == []


# ── norm_moc / norm_lp ────────────────────────────────────────────────────────

def test_norm_moc_prefixes_topic():
    assert norm_moc("DevOps") == "Topic MOC - DevOps"


def test_norm_moc_replaces_ampersand_and_slash():
    assert norm_moc("AI & ML/Ops") == "Topic MOC - AI and ML-Ops"


def test_norm_moc_strips_surrounding_whitespace():
    assert norm_moc("  DevOps  ") == "Topic MOC - DevOps"


def test_norm_lp_prefixes_and_replaces_slash():
    assert norm_lp("Data/Science") == "Learning Path - Data-Science"


def test_norm_lp_keeps_ampersand_unlike_moc():
    assert norm_lp("AI & ML") == "Learning Path - AI & ML"
    assert norm_moc("AI & ML") == "Topic MOC - AI and ML"


# ── safe_stem ─────────────────────────────────────────────────────────────────

def test_safe_stem_strips_extension():
    assert safe_stem("book.pdf") == "book"


def test_safe_stem_without_extension():
    assert safe_stem("README") == "README"


def test_safe_stem_replaces_illegal_filename_chars():
    assert safe_stem('a/b:c*d?.md') == "a_b_c_d_"


# ── load_inventory ────────────────────────────────────────────────────────────

def test_load_inventory_reads_records_and_skips_blank_lines(tmp_path):
    p = tmp_path / "inv.jsonl"
    _seed(
        p,
        json.dumps({"file_name": "a.pdf", "title": "A"}) + "\n"
        + "\n"
        + json.dumps({"file_name": "b.pdf", "title": "B"}) + "\n",
    )
    recs = load_inventory(str(p))
    assert [r["file_name"] for r in recs] == ["a.pdf", "b.pdf"]
    assert recs[1]["title"] == "B"


def test_load_inventory_ignores_whitespace_only_lines(tmp_path):
    p = tmp_path / "inv.jsonl"
    _seed(p, "   \n" + json.dumps({"file_name": "only.pdf"}) + "\n   \n")
    recs = load_inventory(str(p))
    assert len(recs) == 1
    assert recs[0]["file_name"] == "only.pdf"


# ── build_block ───────────────────────────────────────────────────────────────

def test_build_block_wraps_in_markers_and_header():
    block = build_block([("beginner", "Intro", "intro", True)])
    assert block.startswith(BEGIN)
    assert block.endswith(END)
    assert "Resource Notes (1)" in block
    assert "**Beginner**" in block
    assert "- [[intro|Intro]]" in block


def test_build_block_orders_skill_sections_beginner_first():
    block = build_block([
        ("advanced", "A-adv", "a1", True),
        ("beginner", "B-beg", "b1", True),
        ("intermediate", "C-int", "c1", True),
    ])
    beg, inter = block.index("**Beginner**"), block.index("**Intermediate**")
    adv = block.index("**Advanced**")
    assert beg < inter < adv


def test_build_block_sorts_titles_case_insensitively_within_skill():
    block = build_block([
        ("beginner", "banana", "b", True),
        ("beginner", "Apple", "a", True),
    ])
    assert block.index("[[a|Apple]]") < block.index("[[b|banana]]")


def test_build_block_marks_secondary_entries_only():
    block = build_block([
        ("beginner", "Prim", "p", True),
        ("beginner", "Sec", "s", False),
    ])
    assert "[[p|Prim]]" in block
    assert "[[s|Sec]]" in block
    assert block.count("_(secondary)_") == 1


def test_build_block_unknown_skill_bucketed_as_unspecified_last():
    block = build_block([
        ("", "NoLevel", "n", True),
        ("beginner", "Basic", "x", True),
    ])
    assert "**Unspecified**" in block
    assert block.index("**Beginner**") < block.index("**Unspecified**")


# ── inject ────────────────────────────────────────────────────────────────────

def test_inject_appends_block_when_no_markers(tmp_path):
    p = tmp_path / "moc.md"
    _seed(p, "# Title\n\nCurated body.\n")
    block = build_block([("beginner", "Note", "note", True)])
    changed = inject(str(p), block)
    text = _read(p)
    assert changed is True
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    assert "Curated body." in text
    assert "[[note|Note]]" in text


def test_inject_is_idempotent_when_block_unchanged(tmp_path):
    p = tmp_path / "moc.md"
    _seed(p, "# Title\n\nBody.\n")
    block = build_block([("beginner", "Note", "note", True)])
    assert inject(str(p), block) is True
    first = _read(p)
    assert inject(str(p), block) is False
    assert _read(p) == first
    assert _read(p).count(BEGIN) == 1


def test_inject_replaces_block_and_preserves_surrounding_content(tmp_path):
    p = tmp_path / "moc.md"
    stale = BEGIN + "\nSTALE_ENTRY\n" + END
    _seed(p, "# MOC\n\nCURATED_TOP\n\n" + stale + "\n\n## Footer\nCURATED_BOTTOM\n")
    new_block = build_block([("beginner", "Fresh", "fresh", True)])
    changed = inject(str(p), new_block)
    text = _read(p)
    assert changed is True
    assert text.count(BEGIN) == 1
    assert text.count(END) == 1
    assert "STALE_ENTRY" not in text
    assert "[[fresh|Fresh]]" in text
    assert "CURATED_TOP" in text
    assert "CURATED_BOTTOM" in text


# ── main() end-to-end ─────────────────────────────────────────────────────────

def test_main_injects_blocks_and_is_idempotent(tmp_path, capsys):
    gen = tmp_path / "Generated"
    notes = gen / "Resource Notes"
    notes.mkdir(parents=True)
    _seed(notes / "docker-deep-dive.md", "# Docker Deep Dive\n")
    _seed(notes / "k8s-intro.md", "# Kubernetes Intro\n")
    _seed(notes / "ignore.txt", "not a markdown note\n")

    devops_moc = gen / "Topic MOC - DevOps.md"
    devops_lp = gen / "Learning Path - DevOps.md"
    containers_moc = gen / "Topic MOC - Containers.md"
    _seed(devops_moc, "# DevOps MOC\n\nKEEP_DEVOPS_MOC\n")
    _seed(devops_lp, "# DevOps LP\n\nKEEP_DEVOPS_LP\n")
    _seed(containers_moc, "# Containers MOC\n\nKEEP_CONTAINERS\n")

    inv = tmp_path / "inventory.jsonl"
    _seed(
        inv,
        json.dumps({
            "file_name": "docker-deep-dive.pdf", "title": "Docker Deep Dive",
            "skill_level": "intermediate", "primary_topic": "DevOps",
            "secondary_topics": "Containers, Observability",
        }) + "\n"
        + json.dumps({
            "file_name": "k8s-intro.epub", "title": "Kubernetes Intro",
            "skill_level": "beginner", "primary_topic": "DevOps",
            "secondary_topics": "",
        }) + "\n"
        + json.dumps({
            "file_name": "ghost.pdf", "title": "Ghost Resource",
            "skill_level": "beginner", "primary_topic": "DevOps",
            "secondary_topics": "",
        }) + "\n",
    )

    _run_main(inv, gen)
    out = capsys.readouterr().out
    assert "Updated 3 MOC / Learning Path note(s)." in out

    moc_text = _read(devops_moc)
    lp_text = _read(devops_lp)
    cont_text = _read(containers_moc)

    for text in (moc_text, lp_text, cont_text):
        assert text.count(BEGIN) == 1
        assert text.count(END) == 1

    assert "Resource Notes (2)" in moc_text
    assert "[[docker-deep-dive|Docker Deep Dive]]" in moc_text
    assert "[[k8s-intro|Kubernetes Intro]]" in moc_text
    assert "_(secondary)_" not in moc_text
    assert "KEEP_DEVOPS_MOC" in moc_text
    assert "Ghost Resource" not in moc_text

    assert "[[docker-deep-dive|Docker Deep Dive]]" in lp_text
    assert "[[k8s-intro|Kubernetes Intro]]" in lp_text
    assert "_(secondary)_" not in lp_text
    assert "KEEP_DEVOPS_LP" in lp_text

    assert "Resource Notes (1)" in cont_text
    assert "[[docker-deep-dive|Docker Deep Dive]]" in cont_text
    assert "_(secondary)_" in cont_text
    assert "k8s-intro" not in cont_text
    assert "KEEP_CONTAINERS" in cont_text

    before = {p: _read(p) for p in (devops_moc, devops_lp, containers_moc)}
    _run_main(inv, gen)
    out2 = capsys.readouterr().out
    assert "Updated 0 MOC / Learning Path note(s)." in out2
    for p, text in before.items():
        assert _read(p) == text
        assert _read(p).count(BEGIN) == 1


def test_main_missing_notes_subdir_injects_nothing(tmp_path, capsys):
    gen = tmp_path / "Generated"
    gen.mkdir()
    moc = gen / "Topic MOC - DevOps.md"
    original = "# Topic MOC - DevOps\n\nORIGINAL\n"
    _seed(moc, original)
    inv = tmp_path / "inv.jsonl"
    _seed(inv, json.dumps({
        "file_name": "x.pdf", "title": "X", "primary_topic": "DevOps",
    }) + "\n")

    _run_main(inv, gen)
    out = capsys.readouterr().out
    assert "Updated 0 MOC / Learning Path note(s)." in out
    assert BEGIN not in _read(moc)
    assert _read(moc) == original


def test_main_record_without_primary_links_secondary_only(tmp_path):
    gen = tmp_path / "Generated"
    notes = gen / "Resource Notes"
    notes.mkdir(parents=True)
    _seed(notes / "cheatsheet.md", "# Cheatsheet note\n")
    net_moc = gen / "Topic MOC - Networking.md"
    _seed(net_moc, "# Networking MOC\n\nNET_CURATED\n")

    inv = tmp_path / "inv.jsonl"
    _seed(inv, json.dumps({
        "file_name": "cheatsheet.pdf",
        "primary_topic": "", "secondary_topics": "Networking",
    }) + "\n")

    _run_main(inv, gen)
    text = _read(net_moc)
    assert text.count(BEGIN) == 1
    assert "[[cheatsheet|cheatsheet]]" in text
    assert "_(secondary)_" in text
    assert "NET_CURATED" in text


def test_main_dedupes_note_matching_topic_twice(tmp_path):
    gen = tmp_path / "Generated"
    notes = gen / "Resource Notes"
    notes.mkdir(parents=True)
    _seed(notes / "primer.md", "# Primer\n")
    moc = gen / "Topic MOC - Security.md"
    _seed(moc, "# Security MOC\n")

    inv = tmp_path / "inv.jsonl"
    _seed(inv, json.dumps({
        "file_name": "primer.pdf", "title": "Security Primer",
        "skill_level": "beginner", "primary_topic": "Security",
        "secondary_topics": "Security",
    }) + "\n")

    _run_main(inv, gen)
    text = _read(moc)
    assert text.count("[[primer|Security Primer]]") == 1
    assert text.count(BEGIN) == 1
    assert "Resource Notes (1)" in text
