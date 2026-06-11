#!/usr/bin/env python3
"""Generate Obsidian-linked Markdown notes (one per resource) from the inventory
metadata + extracted text.

Each note carries:
  - YAML frontmatter (title, author, type, topics, skill_level, tags, page_count,
    confidence, status, source path, link to the index JSON).
  - A body with a [[wikilink]] to the source file, [[Topic MOC - X]] links for the
    primary/secondary topics that have an existing MOC note, optional
    [[Learning Path - X]] link, inline #tags, and a short text excerpt.

Notes contain only a *summary + excerpt + links* — not the full text (that lives
in the index JSONs). This keeps the Obsidian graph light while staying linked.

Run:
  python3 build_obsidian_notes.py \
      --inventory ".../resource_inventory.jsonl" \
      --text-dir  ".../Books/text_output" \
      --text-dir  ".../Resources/text_output" \
      --generated-dir ".../Career Knowledge Base/Resources/Generated" \
      --index-dir "_catalog/indexed" \
      --out       ".../Career Knowledge Base/Resources/Generated/Resource Notes"
"""
import argparse
import json
import os
import re
import time

EXCERPT_CHARS = 600


def split_list(value):
    if not value:
        return []
    return [p.strip() for p in re.split(r"[;,]", value) if p.strip()]


def to_int(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def norm_moc(topic):
    """Map an inventory topic to its Topic MOC note name."""
    return "Topic MOC - " + topic.replace("&", "and").replace("/", "-").strip()


def norm_lp(topic):
    """Learning Path notes keep '&' but normalise '/'."""
    return "Learning Path - " + topic.replace("/", "-").strip()


def yaml_escape(s):
    s = str(s).replace('"', '\\"')
    return f'"{s}"'


def excerpt(text):
    t = re.sub(r"\s+", " ", text or "").strip()
    return t[:EXCERPT_CHARS] + ("…" if len(t) > EXCERPT_CHARS else "")


def load_inventory(path):
    by_name = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_name[r["file_name"]] = r
    return by_name


def load_extractions(text_dirs):
    by_name = {}
    for d in text_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json") or fn in ("manifest.json", "build_report.json"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("filename"):
                by_name[rec["filename"]] = rec
    return by_name


def existing_notes(generated_dir):
    names = set()
    if os.path.isdir(generated_dir):
        for fn in os.listdir(generated_dir):
            if fn.endswith(".md"):
                names.add(fn[:-3])
    return names


def vault_folder(source_group):
    # Source files live at vault root under Books/ or Resources/.
    return "Books" if source_group == "books" else "Resources"


def icon(source_group):
    return "📖" if source_group == "books" else "📄"


def build_note(inv, ext, moc_set, index_dir):
    title = inv.get("title") or os.path.splitext(inv["file_name"])[0]
    fname = inv["file_name"]
    group = inv.get("source_group", "")
    folder = vault_folder(group)
    source_link = f"[[{folder}/{fname}|{icon(group)} {title}]]"

    primary = inv.get("primary_topic") or ""
    secondary = split_list(inv.get("secondary_topics"))
    tags = split_list(inv.get("tags"))
    skill = inv.get("skill_level") or ""

    # Topic links that resolve to existing MOC notes.
    topic_links = []
    for t in [primary] + secondary:
        if not t:
            continue
        moc = norm_moc(t)
        link = f"[[{moc}]]" if moc in moc_set else f"`{t}`"
        if link not in topic_links:
            topic_links.append(link)
    lp = norm_lp(primary)
    lp_link = f"[[{lp}]]" if lp in moc_set else None

    index_json = os.path.join(index_dir, os.path.splitext(fname)[0] + ".json")

    # ---- frontmatter ----
    fm = ["---"]
    fm.append(f"title: {yaml_escape(title)}")
    if inv.get("author"):
        fm.append(f"author: {yaml_escape(inv['author'])}")
    fm.append(f"type: {inv.get('resource_type') or 'resource'}")
    fm.append(f"source_file: {yaml_escape(folder + '/' + fname)}")
    fm.append(f"index_json: {yaml_escape(index_json)}")
    if primary:
        fm.append(f"primary_topic: {yaml_escape(primary)}")
    if secondary:
        fm.append("secondary_topics: [" + ", ".join(yaml_escape(s) for s in secondary) + "]")
    if skill:
        fm.append(f"skill_level: {skill}")
    pc = to_int(inv.get("page_count"))
    if pc is not None:
        fm.append(f"page_count: {pc}")
    if inv.get("isbn"):
        fm.append(f"isbn: {inv['isbn']}")
    if inv.get("language"):
        fm.append(f"language: {inv['language']}")
    fm.append(f"confidence: {inv.get('confidence') or 'unknown'}")
    fm.append(f"status: {inv.get('status') or 'classified'}")
    # tags: granular tags + a slugified skill level
    all_tags = list(tags)
    if skill:
        all_tags.append(skill)
    all_tags.append("resource-note")
    if all_tags:
        fm.append("tags: [" + ", ".join(t.replace(" ", "-") for t in all_tags) + "]")
    today = time.strftime("%Y-%m-%d")
    fm.append(f"created: {today}")
    fm.append(f"updated: {today}")
    fm.append("---")

    # ---- body ----
    body = [f"\n# {title}\n"]
    meta_bits = []
    if inv.get("resource_type"):
        meta_bits.append(f"`{inv['resource_type']}`")
    if skill:
        meta_bits.append(f"**{skill}**")
    if pc:
        meta_bits.append(f"{pc}p")
    body.append("> " + " · ".join(meta_bits) if meta_bits else "")
    body.append("")
    body.append(f"**Source:** {source_link}")
    if topic_links:
        body.append(f"**Topics:** {' · '.join(topic_links)}")
    if lp_link:
        body.append(f"**Learning path:** {lp_link}")
    if tags:
        body.append("**Tags:** " + " ".join(f"#{t.replace(' ', '-')}" for t in tags))

    ex = ext.get("extraction") if isinstance(ext.get("extraction"), dict) else None
    e = ext
    ocr = e.get("ocr_used")
    chars = e.get("total_chars")
    stat_bits = []
    if chars:
        stat_bits.append(f"{chars:,} chars extracted")
    if ocr:
        stat_bits.append(f"OCR on {e.get('ocr_page_count')} page(s)")
    if stat_bits:
        body.append("")
        body.append("> [!info] Extraction\n> " + " · ".join(stat_bits))

    body.append("")
    body.append("## Excerpt")
    body.append("")
    body.append(excerpt(e.get("text", "")))
    body.append("")

    return "\n".join(fm) + "\n".join(body)


def safe_stem(file_name):
    stem = os.path.splitext(file_name)[0]
    return re.sub(r'[\\/:*?"<>|]', "_", stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--text-dir", action="append", required=True, dest="text_dirs")
    ap.add_argument("--generated-dir", required=True,
                    help="Folder holding the existing Topic MOC / Learning Path notes")
    ap.add_argument("--index-dir", default="_catalog/indexed",
                    help="Vault-relative path where the index JSONs live (for frontmatter)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    inv = load_inventory(args.inventory)
    ext = load_extractions(args.text_dirs)
    moc_set = existing_notes(args.generated_dir)

    written, skipped_no_moc_topics = 0, set()
    for name, erec in sorted(ext.items()):
        irec = inv.get(name)
        if irec is None:
            continue
        note = build_note(irec, erec, moc_set, args.index_dir)
        out_path = os.path.join(args.out, safe_stem(name) + ".md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(note)
        written += 1
        for t in [irec.get("primary_topic")] + split_list(irec.get("secondary_topics")):
            if t and norm_moc(t) not in moc_set:
                skipped_no_moc_topics.add(t)

    print(f"Wrote {written} Obsidian notes -> {args.out}")
    if skipped_no_moc_topics:
        print(f"  topics without a MOC note (kept as tags): {sorted(skipped_no_moc_topics)}")


if __name__ == "__main__":
    main()
