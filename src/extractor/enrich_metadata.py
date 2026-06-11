#!/usr/bin/env python3
"""Enrich the classification inventory with metadata recovered from the source
files themselves: embedded PDF/EPUB title & author, EPUB language, and ISBNs
parsed from the extracted text.

Conservative by design:
  - Only FILLS a field when it is currently empty, or (for title) when the title
    is just the filename stem. Never overwrites a good existing value.
  - Validates embedded values and rejects junk ("untitled", software names,
    filename-like titles, etc.).

The original inventory is left untouched; an enriched copy is written.

Run:
  python3 enrich_metadata.py \
      --inventory ".../resource_inventory.jsonl" \
      --source-dir ".../Books" --source-dir ".../Resources" \
      --text-dir ".../text_output_books" --text-dir ".../text_output_resources" \
      --out ".../resource_inventory_enriched.jsonl"
"""
import argparse
import json
import os
import re
import zipfile

import fitz  # PyMuPDF

JUNK_TITLES = {"untitled", "title", "unknown", "microsoft word", "document", ""}
SOFTWARE_HINTS = re.compile(
    r"(adobe|acrobat|calibre|microsoft|pscript|ghostscript|quark|indesign|"
    r"latex|pdftk|word|writer|nitro)", re.I)
ISBN13 = re.compile(r"97[89](?:[-\s]?\d){10}")
ISBN10 = re.compile(r"\b\d(?:[-\s]?\d){8}[-\s]?[\dXx]\b")


def clean_title(value, stem):
    if not value:
        return None
    t = value.strip()
    low = t.lower()
    if low in JUNK_TITLES or len(t) < 3:
        return None
    if "microsoft word" in low or re.search(r"\.(doc|docx|indd|pdf|tex|qxd)\b", low):
        return None
    if t.isdigit():
        return None
    if low == stem.lower():
        return None
    # Collapse whitespace.
    return re.sub(r"\s+", " ", t)


def clean_author(value, title):
    if not value:
        return None
    a = value.strip()
    if len(a) < 2 or a.isdigit():
        return None
    if SOFTWARE_HINTS.search(a):
        return None
    if title and a.strip().lower() == title.strip().lower():
        return None
    return re.sub(r"\s+", " ", a)


def normalize_isbn(s):
    return re.sub(r"[-\s]", "", s)


def find_isbn(text):
    if not text:
        return None
    head = text[:8000] + "\n" + text[-8000:]
    m = ISBN13.search(head)
    if m:
        return normalize_isbn(m.group(0))
    m = ISBN10.search(head)
    if m:
        v = normalize_isbn(m.group(0))
        if len(v) == 10:
            return v
    return None


def pdf_meta(path):
    try:
        d = fitz.open(path)
        m = d.metadata or {}
        d.close()
        return m.get("title"), m.get("author"), None
    except Exception:
        return None, None, None


def epub_meta(path):
    try:
        with zipfile.ZipFile(path) as z:
            opf_path = None
            if "META-INF/container.xml" in z.namelist():
                cx = z.read("META-INF/container.xml").decode("utf-8", "ignore")
                mm = re.search(r'full-path="([^"]+)"', cx)
                if mm:
                    opf_path = mm.group(1)
            if not opf_path:
                return None, None, None
            opf = z.read(opf_path).decode("utf-8", "ignore")
            def dc(tag):
                mm = re.search(rf"<dc:{tag}[^>]*>(.*?)</dc:{tag}>", opf, re.I | re.S)
                return re.sub(r"<[^>]+>", "", mm.group(1)).strip() if mm else None
            return dc("title"), dc("creator"), dc("language")
    except Exception:
        return None, None, None


def load_text_index(text_dirs):
    idx = {}
    for d in text_dirs:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.endswith(".json") and fn not in ("manifest.json", "build_report.json"):
                idx[fn] = os.path.join(d, fn)
    return idx


def get_text(text_index, file_name):
    key = os.path.splitext(file_name)[0] + ".json"
    path = text_index.get(key)
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("text", "")
    except Exception:
        return ""


def find_source(source_dirs, file_name):
    for d in source_dirs:
        p = os.path.join(d, file_name)
        if os.path.isfile(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--source-dir", action="append", required=True, dest="source_dirs")
    ap.add_argument("--text-dir", action="append", default=[], dest="text_dirs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    text_index = load_text_index(args.text_dirs)
    stats = {"title": 0, "author": 0, "language": 0, "isbn": 0, "missing_source": 0}
    out_records = []
    changes = []

    with open(args.inventory, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fname = r["file_name"]
            stem = os.path.splitext(fname)[0]
            ftype = (r.get("file_type") or "").lower()
            src = find_source(args.source_dirs, fname)

            emb_title = emb_author = emb_lang = None
            if src:
                if ftype == "epub" or fname.lower().endswith(".epub"):
                    emb_title, emb_author, emb_lang = epub_meta(src)
                elif ftype == "pdf" or fname.lower().endswith(".pdf"):
                    emb_title, emb_author, emb_lang = pdf_meta(src)
            else:
                stats["missing_source"] += 1

            changed = {}
            cur_title = (r.get("title") or "").strip()
            ct = clean_title(emb_title, stem)
            if ct and (not cur_title or cur_title == stem):
                r["title"] = ct
                changed["title"] = ct
                stats["title"] += 1

            ca = clean_author(emb_author, r.get("title"))
            if ca and not (r.get("author") or "").strip():
                r["author"] = ca
                changed["author"] = ca
                stats["author"] += 1

            if emb_lang and not (r.get("language") or "").strip():
                r["language"] = emb_lang.strip()
                changed["language"] = emb_lang.strip()
                stats["language"] += 1

            if not (r.get("isbn") or "").strip() if "isbn" in r else True:
                isbn = find_isbn(get_text(text_index, fname))
                if isbn:
                    r["isbn"] = isbn
                    changed["isbn"] = isbn
                    stats["isbn"] += 1

            r["enriched"] = bool(changed)
            out_records.append(r)
            if changed:
                changes.append({"file_name": fname, **changed})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = os.path.splitext(args.out)[0] + "_changes.json"
    with open(report, encoding="utf-8", mode="w") as f:
        json.dump({"stats": stats, "changes": changes}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out_records)} records -> {args.out}")
    print(f"  filled: title={stats['title']} author={stats['author']} "
          f"language={stats['language']} isbn={stats['isbn']}")
    if stats["missing_source"]:
        print(f"  {stats['missing_source']} records had no source file on disk")
    print(f"  change log -> {report}")


if __name__ == "__main__":
    main()
