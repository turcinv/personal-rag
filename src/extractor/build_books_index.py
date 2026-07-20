#!/usr/bin/env python3
"""Regenerate the vault's single "Books Index" aggregate note from the
enriched resource inventory.

This is the missing Phase 5 step called out in
`Templates/Book Catalog Sync Agent Prompt.md` in the vault repo: every other
build step (index, notes, sqlite, MOC backlinks) has a dedicated script, but
`Resources/Generated/Books Index.md` was previously hand-edited, so it drifts
out of sync with the catalog whenever new books are enriched without a
matching manual pass.

Only rows with resource_type == "book" are included — this note is an
aggregate index, not a per-book note (the vault's no-per-book-notes rule is
about avoiding one Markdown file per book; this is exactly the one
consolidated exception the rule allows, same as the existing Resource Notes
pipeline).

Run:
  python3 build_books_index.py \
      --inventory ".../resource_inventory_enriched.jsonl" \
      --out       ".../Career Knowledge Base/Resources/Generated/Books Index.md"
"""
import argparse
import json
import os
import re
import time

FRONTMATTER_DEFAULTS = {
    "domain": "Personal Knowledge",
    "type": "Reference",
    "tags": "[books, index, catalog, resources]",
    "source": "Resource Catalog",
    "confidence": "high",
    "status": "processed",
}


def split_list(value):
    if not value:
        return []
    return [p.strip() for p in re.split(r"[;,]", value) if p.strip()]


def to_int(value):
    try:
        n = int(str(value).strip())
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def load_books(inventory_path):
    books = []
    with open(inventory_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("resource_type") == "book":
                books.append(r)
    return books


def existing_created_date(out_path):
    """Preserve the note's original `created:` date across regenerations."""
    if not os.path.isfile(out_path):
        return None
    with open(out_path, encoding="utf-8") as f:
        head = f.read(2000)
    m = re.search(r"^created:\s*(\S+)", head, re.MULTILINE)
    return m.group(1) if m else None


def format_entry(book):
    title = book.get("title") or os.path.splitext(book["file_name"])[0]
    stem = os.path.splitext(book["file_name"])[0]
    skill = (book.get("skill_level") or "Unknown").strip()
    skill_display = skill[:1].upper() + skill[1:] if skill else "Unknown"
    pages = to_int(book.get("page_count"))
    pages_display = f"{pages}p" if pages else "—"

    lines = [f"- **{title}** · {skill_display} · {pages_display}"]
    lines.append(f"  [[Books/{book['file_name']}|📖 {stem}]]")
    tags = split_list(book.get("tags"))
    if tags:
        lines.append("  *" + ", ".join(tags) + "*")
    return "\n".join(lines)


def build_markdown(books, out_path):
    by_topic = {}
    for b in books:
        topic = b.get("primary_topic") or "Other"
        by_topic.setdefault(topic, []).append(b)

    # Descending count, ties broken by ASCII-ascending topic name — matches
    # the ordering already established in the hand-authored note.
    topics_sorted = sorted(by_topic.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    created = existing_created_date(out_path) or time.strftime("%Y-%m-%d")
    today = time.strftime("%Y-%m-%d")

    lines = ["---"]
    lines.append("title: Books Index")
    lines.append(f"domain: {FRONTMATTER_DEFAULTS['domain']}")
    lines.append(f"type: {FRONTMATTER_DEFAULTS['type']}")
    lines.append(f"created: {created}")
    lines.append(f"updated: {today}")
    lines.append(f"tags: {FRONTMATTER_DEFAULTS['tags']}")
    lines.append(f"source: {FRONTMATTER_DEFAULTS['source']}")
    lines.append(f"confidence: {FRONTMATTER_DEFAULTS['confidence']}")
    lines.append(f"status: {FRONTMATTER_DEFAULTS['status']}")
    lines.append("---")
    lines.append("")
    lines.append("# Books Index")
    lines.append("")
    lines.append(f"> {len(books)} books catalogued across {len(by_topic)} topics.")
    lines.append("> Source: `Resources/_catalog/resource_inventory.csv`")
    lines.append("> Vault location: `Books/` — files are accessible in Obsidian.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Topic                   | Count |")
    lines.append("| ----------------------- | ----- |")
    for topic, group in topics_sorted:
        lines.append(f"| {topic:<24} | {len(group):<5} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    for topic, group in topics_sorted:
        lines.append(f"## {topic}")
        lines.append("")
        for book in sorted(group, key=lambda b: (b.get("title") or b["file_name"]).lower()):
            lines.append(format_entry(book))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True,
                     help="Path to resource_inventory_enriched.jsonl")
    ap.add_argument("--out", required=True,
                     help="Path to write Resources/Generated/Books Index.md")
    args = ap.parse_args()

    books = load_books(args.inventory)
    if not books:
        raise SystemExit(f"No resource_type=='book' rows found in {args.inventory}")

    markdown = build_markdown(books, args.out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)

    topics = {b.get("primary_topic") or "Other" for b in books}
    print(f"Wrote Books Index -> {args.out}")
    print(f"  {len(books)} books across {len(topics)} topics")


if __name__ == "__main__":
    main()
