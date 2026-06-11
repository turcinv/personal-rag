#!/usr/bin/env python3
"""Analyze files in a directory: detect file type and whether text can be extracted.

Pre-flight survey before running extract_text. Writes file_analysis.csv.
"""
import argparse
import csv
import os
import re
import zipfile

import fitz  # PyMuPDF


def detect_type(path):
    """Detect file type from magic bytes."""
    with open(path, "rb") as f:
        head = f.read(8)
    if head.startswith(b"%PDF"):
        return "PDF"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                if "mimetype" in z.namelist():
                    mt = z.read("mimetype").decode("ascii", "ignore").strip()
                    if mt == "application/epub+zip":
                        return "EPUB"
                return "ZIP"
        except zipfile.BadZipFile:
            return "ZIP(corrupt)"
    return "UNKNOWN"


def pdf_has_text(path, sample_pages=12, min_chars=100):
    """Return (extractable, note). Samples pages to see if real text exists."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        return False, f"open error: {e}"
    n = doc.page_count
    if n == 0:
        return False, "no pages"
    idxs = sorted(set(int(i * (n - 1) / (sample_pages - 1)) for i in range(sample_pages))) if n > 1 else [0]
    total = 0
    for i in idxs:
        try:
            total += len(doc.load_page(i).get_text("text").strip())
        except Exception:
            pass
    doc.close()
    if total >= min_chars:
        return True, f"{total} chars in {len(idxs)} sampled pages of {n}"
    return False, f"only {total} chars in sample (likely scanned/image)"


def epub_has_text(path, min_chars=100):
    """EPUBs store XHTML; check for text content."""
    try:
        with zipfile.ZipFile(path) as z:
            html_files = [n for n in z.namelist() if n.lower().endswith((".html", ".xhtml", ".htm"))]
            total = 0
            for name in html_files[:20]:
                raw = z.read(name).decode("utf-8", "ignore")
                text = re.sub(r"<[^>]+>", " ", raw)
                total += len(text.strip())
                if total >= min_chars:
                    return True, f"text found in xhtml ({len(html_files)} docs)"
            return (total >= min_chars), f"{total} chars sampled in {len(html_files)} docs"
    except Exception as e:
        return False, f"error: {e}"


def main():
    ap = argparse.ArgumentParser(description="Pre-flight survey of a document directory")
    ap.add_argument("dir", nargs="?", default=None,
                    help="Directory to scan (default: current working directory)")
    args = ap.parse_args()

    DIR = os.path.abspath(args.dir) if args.dir else os.getcwd()

    files = sorted(
        f for f in os.listdir(DIR)
        if os.path.isfile(os.path.join(DIR, f)) and not f.startswith(".")
        and not f.endswith((".csv", ".py"))
    )
    rows = []
    for fname in files:
        path = os.path.join(DIR, fname)
        ftype = detect_type(path)
        if ftype == "PDF":
            ok, note = pdf_has_text(path)
        elif ftype == "EPUB":
            ok, note = epub_has_text(path)
        else:
            ok, note = False, "unsupported type"
        rows.append((fname, ftype, "Yes" if ok else "No", note))
        print(f"[{ftype:5}] {'OK ' if ok else 'NO '} {fname}")

    out = os.path.join(DIR, "file_analysis.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "file_type", "text_extractable", "notes"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
