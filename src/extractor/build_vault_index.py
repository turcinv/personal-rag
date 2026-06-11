#!/usr/bin/env python3
"""Index Obsidian vault knowledge notes into a JSONL file compatible with
build_sqlite.py's input format.

Walks the Knowledge/ folder of an Obsidian vault, parses YAML frontmatter,
and emits one JSON record per note into vault_documents.jsonl in the target
index directory.  The resulting JSONL can be fed to build_sqlite.py alongside
index_documents.jsonl so that books/resources and vault notes live in the same
FTS database and are searchable together.

Field mapping  (vault frontmatter → indexed document):
  domain      → primary_topic
  type        → resource_type
  tags        → tags
  status      → classification_status
  updated     → classified_at
  confidence  → confidence
  (body text) → text
  source_group = "vault" (fixed)

Run:
  python3 build_vault_index.py \
      --vault  "~/…/Career Knowledge Base" \
      --out    "~/Documents/knowledge-base-index/indexed"
"""
import argparse
import hashlib
import json
import os
import re
import sys


def parse_frontmatter(text):
    """Return (metadata_dict, body_str) from a Markdown file.  If the file has
    no YAML frontmatter the entire text is treated as body."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    raw_fm = text[4:end]
    body = text[end + 4:].strip()
    meta = {}
    # Simple line-by-line YAML parser (handles scalar values and list blocks).
    current_key = None
    list_buf = []
    for line in raw_fm.splitlines():
        # Indented list item
        if line.startswith("  - ") or line.startswith("- "):
            item = re.sub(r"^\s*-\s*", "", line).strip().strip('"').strip("'")
            if item:
                list_buf.append(item)
            continue
        # Key: value line
        m = re.match(r"^(\w[\w -]*):\s*(.*)", line)
        if m:
            if current_key and list_buf:
                meta[current_key] = list_buf
                list_buf = []
            current_key = m.group(1).strip()
            val = m.group(2).strip().strip('"').strip("'")
            if val:
                meta[current_key] = val
            else:
                list_buf = []
        elif line.strip() == "" and current_key and list_buf:
            meta[current_key] = list_buf
            list_buf = []
            current_key = None
    if current_key and list_buf:
        meta[current_key] = list_buf
    return meta, body


def norm_tags(raw):
    """Return a clean list of tag strings from frontmatter value (list or
    comma-separated string)."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip().lower().lstrip("#") for t in raw if str(t).strip()]
    return [t.strip().lower().lstrip("#") for t in re.split(r"[,\s]+", str(raw)) if t.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="Index Obsidian vault notes into vault_documents.jsonl"
    )
    ap.add_argument("--vault", required=True,
                    help="Path to the Obsidian vault / git repo root "
                         "(e.g. ~/…/Career Knowledge Base)")
    ap.add_argument("--knowledge-dir", default="Knowledge",
                    help="Subfolder within the vault to walk (default: Knowledge)")
    ap.add_argument("--out", required=True,
                    help="Output directory (vault_documents.jsonl is written here)")
    args = ap.parse_args()

    vault_root = os.path.realpath(os.path.expanduser(args.vault))
    knowledge_root = os.path.join(vault_root, args.knowledge_dir)
    if not os.path.isdir(knowledge_root):
        sys.exit(f"Knowledge folder not found: {knowledge_root}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "vault_documents.jsonl")

    records = []
    skipped_moc = skipped_empty = 0

    for dirpath, _, filenames in os.walk(knowledge_root):
        for fname in sorted(filenames):
            if not fname.endswith(".md"):
                continue
            full_path = os.path.join(dirpath, fname)
            # Vault-relative path used as file_name key.
            rel_path = os.path.relpath(full_path, vault_root).replace(os.sep, "/")

            try:
                text = open(full_path, encoding="utf-8").read()
            except Exception:
                continue

            meta, body = parse_frontmatter(text)

            # Skip MOC files (navigation-only, not knowledge content)
            if "MOC" in fname or meta.get("type", "") == "MOC":
                skipped_moc += 1
                continue

            # Skip empty / near-empty bodies
            word_count = len(body.split())
            if word_count < 10:
                skipped_empty += 1
                continue

            title = meta.get("title", "") or fname[:-3]
            domain = meta.get("domain", "")
            tags = norm_tags(meta.get("tags", []))
            doc_id = hashlib.sha256(rel_path.encode()).hexdigest()
            file_size = os.path.getsize(full_path)
            updated = meta.get("updated", "") or meta.get("created", "")
            if updated and len(updated) == 10:
                updated = updated + "T00:00:00Z"

            record = {
                "id": doc_id,
                "file_name": rel_path,
                "file_type": "md",
                "source_group": "vault",
                "source_bucket": "",
                "gcs_path": "",
                "file_size_bytes": file_size,
                "title": title,
                "author": "",
                "language": "en",
                "isbn": None,
                "page_count": None,
                "resource_type": meta.get("type", "Knowledge"),
                "primary_topic": domain,
                "secondary_topics": [],
                "skill_level": "",
                "tags": tags,
                "confidence": meta.get("confidence", ""),
                "classification_status": meta.get("status", ""),
                "classified_at": updated,
                "extraction": {
                    "extracted_at": updated,
                    "total_pages": None,
                    "total_documents": None,
                    "text_layer_chars": len(body),
                    "ocr_used": False,
                    "ocr_pages": [],
                    "ocr_page_count": 0,
                    "ocr_chars": 0,
                    "total_chars": len(body),
                },
                "text": body,
            }
            records.append(record)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} vault notes  →  {out_path}")
    print(f"  skipped: {skipped_moc} MOC files, {skipped_empty} empty/stub notes")


if __name__ == "__main__":
    main()
