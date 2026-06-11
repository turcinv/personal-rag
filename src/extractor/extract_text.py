#!/usr/bin/env python3
"""Extract all text from every PDF/EPUB in a directory.

Strategy:
  - PDFs: use the embedded text layer per page (PyMuPDF). Pages that have
    essentially no extractable text (image/scanned pages) are rendered to an
    image and run through Tesseract OCR. Fully text-based files never invoke OCR.
  - EPUBs: concatenate text from the XHTML documents in spine order.

Output: one JSON file per book in text_output/, containing metadata + full text.
A manifest.json summarizes the whole run.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from html import unescape

import fitz  # PyMuPDF

OCR_CHAR_THRESHOLD = 5
OCR_DPI = 300
OCR_LANG = "eng"

# Module-level log path, set in main() before any calls to log().
_log_path = None


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_path:
        with open(_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def detect_type(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown"):
        return "MD"
    if ext == ".txt":
        return "TXT"
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


def ocr_image_bytes(png_bytes):
    """Run tesseract on PNG bytes via stdin/stdout. Returns extracted text."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        tf.write(png_bytes)
        tmp = tf.name
    try:
        res = subprocess.run(
            ["tesseract", tmp, "stdout", "-l", OCR_LANG],
            capture_output=True, text=True,
        )
        return res.stdout
    finally:
        os.unlink(tmp)


def extract_pdf(path):
    doc = fitz.open(path)
    n = doc.page_count
    parts = []
    ocr_pages = []
    text_layer_chars = 0
    ocr_chars = 0
    mat = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
    for i in range(n):
        page = doc.load_page(i)
        text = page.get_text("text")
        stripped = text.strip()
        if len(stripped) >= OCR_CHAR_THRESHOLD:
            text_layer_chars += len(stripped)
            parts.append(text)
        else:
            try:
                pix = page.get_pixmap(matrix=mat)
                ocr_text = ocr_image_bytes(pix.tobytes("png"))
            except Exception as e:
                ocr_text = ""
                log(f"    ! OCR failed on page {i+1}: {e}")
            ocr_text = ocr_text.strip()
            if ocr_text:
                ocr_pages.append(i + 1)
                ocr_chars += len(ocr_text)
                parts.append(ocr_text)
    doc.close()
    full = "\n\n".join(parts)
    meta = {
        "total_pages": n,
        "text_layer_chars": text_layer_chars,
        "ocr_pages": ocr_pages,
        "ocr_page_count": len(ocr_pages),
        "ocr_chars": ocr_chars,
        "total_chars": len(full),
        "ocr_used": bool(ocr_pages),
    }
    return full, meta


def _strip_html(raw):
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?is)<br\s*/?>", "\n", raw)
    raw = re.sub(r"(?is)</(p|div|h[1-6]|li|tr)>", "\n", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _epub_spine_order(z):
    """Return content document names in spine order, falling back to all xhtml."""
    try:
        opf_path = None
        if "META-INF/container.xml" in z.namelist():
            cx = z.read("META-INF/container.xml").decode("utf-8", "ignore")
            m = re.search(r'full-path="([^"]+)"', cx)
            if m:
                opf_path = m.group(1)
        if opf_path and opf_path in z.namelist():
            opf = z.read(opf_path).decode("utf-8", "ignore")
            base = os.path.dirname(opf_path)
            ids = dict(re.findall(r'<item\s[^>]*id="([^"]+)"[^>]*href="([^"]+)"', opf))
            for m2 in re.finditer(r'<item\s[^>]*href="([^"]+)"[^>]*id="([^"]+)"', opf):
                ids[m2.group(2)] = m2.group(1)
            order = []
            for ref in re.findall(r'<itemref\s[^>]*idref="([^"]+)"', opf):
                href = ids.get(ref)
                if href:
                    full = os.path.normpath(os.path.join(base, href)).replace(os.sep, "/")
                    if full in z.namelist():
                        order.append(full)
            if order:
                return order
    except Exception:
        pass
    return [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]


def extract_epub(path):
    parts = []
    with zipfile.ZipFile(path) as z:
        docs = _epub_spine_order(z)
        for name in docs:
            try:
                raw = z.read(name).decode("utf-8", "ignore")
            except Exception:
                continue
            t = _strip_html(raw)
            if t:
                parts.append(t)
    full = "\n\n".join(parts)
    meta = {
        "total_documents": len(parts),
        "text_layer_chars": len(full),
        "ocr_pages": [],
        "ocr_page_count": 0,
        "ocr_chars": 0,
        "total_chars": len(full),
        "ocr_used": False,
    }
    return full, meta


def extract_md(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        full = f.read()
    meta = {
        "total_documents": 1,
        "text_layer_chars": len(full),
        "ocr_pages": [],
        "ocr_page_count": 0,
        "ocr_chars": 0,
        "total_chars": len(full),
        "ocr_used": False,
    }
    return full, meta


def main():
    global _log_path

    # First CLI arg may be the target directory.
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        DIR = os.path.abspath(sys.argv[1])
        sys.argv.pop(1)
    elif len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print("Usage: rag-extract [directory] [filename]")
        sys.exit(0)
    else:
        DIR = os.getcwd()

    OUT_DIR = os.path.join(DIR, "text_output")
    _log_path = os.path.join(DIR, "extract_progress.log")

    os.makedirs(OUT_DIR, exist_ok=True)
    open(_log_path, "w").close()

    skip = {"extract_text.py", "analyze_files.py"}
    files = sorted(
        f for f in os.listdir(DIR)
        if os.path.isfile(os.path.join(DIR, f)) and not f.startswith(".")
        and f not in skip and not f.endswith((".csv", ".log", ".py"))
    )

    # Optional second arg: process only one named file.
    if len(sys.argv) > 1:
        target = sys.argv[1]
        files = [f for f in files if f == target or f == os.path.basename(target)]
        if not files:
            log(f"No matching file for '{sys.argv[1]}'")
            return

    manifest = []
    total = len(files)
    log(f"Starting extraction of {total} file(s) from {DIR}")
    for idx, fname in enumerate(files, 1):
        path = os.path.join(DIR, fname)
        out_name = os.path.splitext(fname)[0] + ".json"
        out_path = os.path.join(OUT_DIR, out_name)
        if os.path.exists(out_path):
            log(f"({idx}/{total}) SKIP already extracted: {fname}")
            manifest.append({"filename": fname, "status": "cached"})
            continue
        ftype = detect_type(path)
        t0 = time.time()
        try:
            if ftype == "PDF":
                text, meta = extract_pdf(path)
            elif ftype == "EPUB":
                text, meta = extract_epub(path)
            elif ftype in ("MD", "TXT"):
                text, meta = extract_md(path)
            else:
                log(f"({idx}/{total}) SKIP unsupported {ftype}: {fname}")
                manifest.append({"filename": fname, "file_type": ftype, "status": "skipped"})
                continue
        except Exception as e:
            log(f"({idx}/{total}) ERROR {fname}: {e}")
            manifest.append({"filename": fname, "file_type": ftype, "status": "error", "error": str(e)})
            continue

        record = {
            "filename": fname,
            "file_type": ftype,
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **meta,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({**record, "text": text}, f, ensure_ascii=False, indent=2)

        record["status"] = "ok"
        record["output"] = os.path.join("text_output", out_name)
        manifest.append(record)
        ocr_note = f" OCR:{meta['ocr_page_count']}pg/{meta['ocr_chars']}ch" if meta["ocr_used"] else ""
        log(f"({idx}/{total}) OK {ftype} {fname} -> {meta['total_chars']} chars{ocr_note} ({time.time()-t0:.1f}s)")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ocr_files = [m for m in manifest if m.get("ocr_used")]
    log(f"DONE. {len(manifest)} files. OCR used on {len(ocr_files)} file(s).")


if __name__ == "__main__":
    main()
