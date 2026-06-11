#!/usr/bin/env python3
"""Join classification metadata with extracted full text into index-ready JSONs.

Inputs:
  - An inventory JSONL (resource_inventory.jsonl) with one classification record
    per resource, keyed by `file_name`.
  - One or more `text_output/` directories (produced by extract_text.py) holding
    per-file extraction JSONs with the full `text` and extraction stats.

Output (one per matched document):
  - <out_dir>/<file_stem>.json      : merged metadata + full text
  - <out_dir>/index_documents.jsonl : all merged records, one per line (no indent)
  - <out_dir>/build_report.json     : what matched / what didn't

Run:
  python3 build_index_documents.py \
      --inventory ".../resource_inventory.jsonl" \
      --text-dir  ".../Books/text_output" \
      --text-dir  ".../Resources/text_output" \
      --out       ".../_catalog/indexed"
"""
import argparse
import json
import os
import re
import time


def split_list(value):
    """Inventory stores tags/topics as comma-separated strings -> list."""
    if not value:
        return []
    return [p.strip() for p in re.split(r"[;,]", value) if p.strip()]


def to_int(value):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return None


def load_inventory(path):
    by_name = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_name[r["file_name"]] = r
    return by_name


def load_extractions(text_dirs):
    """Map file_name -> (extraction_record, source_dir)."""
    by_name = {}
    for d in text_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".json") or fn in ("manifest.json", "build_report.json"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                rec = json.load(f)
            name = rec.get("filename")
            if name:
                by_name[name] = (rec, d)
    return by_name


def merge(inv, ext):
    """Build one index-ready record from inventory + extraction records."""
    return {
        "id": inv.get("id"),
        "file_name": inv.get("file_name"),
        "file_type": inv.get("file_type"),
        "source_group": inv.get("source_group"),
        "source_bucket": inv.get("bucket"),
        "gcs_path": inv.get("gcs_path"),
        "file_size_bytes": to_int(inv.get("file_size")),
        # --- classification metadata ---
        "title": inv.get("title") or None,
        "author": inv.get("author") or None,
        "language": inv.get("language") or None,
        "isbn": inv.get("isbn") or None,
        "page_count": to_int(inv.get("page_count")),
        "resource_type": inv.get("resource_type") or None,
        "primary_topic": inv.get("primary_topic") or None,
        "secondary_topics": split_list(inv.get("secondary_topics")),
        "skill_level": inv.get("skill_level") or None,
        "tags": split_list(inv.get("tags")),
        "confidence": inv.get("confidence") or None,
        "classification_status": inv.get("status") or None,
        "classified_at": inv.get("created_at") or None,
        # --- extraction stats ---
        "extraction": {
            "extracted_at": ext.get("extracted_at"),
            "total_pages": ext.get("total_pages"),
            "total_documents": ext.get("total_documents"),
            "text_layer_chars": ext.get("text_layer_chars"),
            "ocr_used": ext.get("ocr_used"),
            "ocr_pages": ext.get("ocr_pages", []),
            "ocr_page_count": ext.get("ocr_page_count", 0),
            "ocr_chars": ext.get("ocr_chars", 0),
            "total_chars": ext.get("total_chars"),
        },
        # --- the indexable content ---
        "text": ext.get("text", ""),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--text-dir", action="append", required=True, dest="text_dirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    inv = load_inventory(args.inventory)
    ext = load_extractions(args.text_dirs)

    matched, jsonl_records = [], []
    for name, (erec, _) in sorted(ext.items()):
        irec = inv.get(name)
        if irec is None:
            continue
        rec = merge(irec, erec)
        out_name = os.path.splitext(name)[0] + ".json"
        with open(os.path.join(args.out, out_name), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        matched.append(name)
        jsonl_records.append(rec)

    # Combined JSONL for bulk indexing (one compact record per line).
    with open(os.path.join(args.out, "index_documents.jsonl"), "w", encoding="utf-8") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    inv_only = sorted(set(inv) - set(ext))   # in inventory, no extracted text
    ext_only = sorted(set(ext) - set(inv))   # extracted, but not in inventory
    report = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inventory_records": len(inv),
        "extraction_records": len(ext),
        "matched": len(matched),
        "in_inventory_without_text": inv_only,
        "extracted_without_inventory": ext_only,
        "out_dir": args.out,
    }
    with open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Matched & wrote {len(matched)} index documents -> {args.out}")
    print(f"  inventory={len(inv)} extraction={len(ext)}")
    if inv_only:
        print(f"  {len(inv_only)} in inventory without extracted text: {inv_only}")
    if ext_only:
        print(f"  {len(ext_only)} extracted without inventory match: {ext_only}")


if __name__ == "__main__":
    main()
