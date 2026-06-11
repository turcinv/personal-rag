#!/usr/bin/env python3
"""Find near-duplicate documents by content (and exact ISBN match).

Signals:
  - Same ISBN  -> almost certainly the same book (different filename/scan).
  - High Jaccard similarity of word-shingles over a text sample -> same or
    closely related content (e.g. different editions, re-uploads).

Outputs a Markdown report with candidate pairs, grouped and sorted by strength,
each linking to the source files via [[wikilinks]] for review in Obsidian.

Run:
  python3 dup_detect.py \
      --inventory ".../resource_inventory_enriched.jsonl" \
      --text-dir ".../text_output_books" --text-dir ".../text_output_resources" \
      --out ".../Generated/Content Duplicate Candidates.md" \
      --threshold 0.35
"""
import argparse
import json
import os
import re

SHINGLE_K = 5
SAMPLE_CHARS = 20000
WORD = re.compile(r"[a-z0-9]+")


def shingles(text):
    words = WORD.findall((text or "")[:SAMPLE_CHARS].lower())
    if len(words) < SHINGLE_K:
        return set()
    return {" ".join(words[i:i + SHINGLE_K]) for i in range(len(words) - SHINGLE_K + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def load_inventory(path):
    by_name = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_name[r["file_name"]] = r
    return by_name


def load_texts(text_dirs):
    out = {}
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
                out[name] = rec.get("text", "")
    return out


def vault_link(inv, name):
    folder = "Books" if (inv.get(name, {}).get("source_group") == "books") else "Resources"
    title = inv.get(name, {}).get("title") or os.path.splitext(name)[0]
    return f"[[{folder}/{name}|{title}]]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--text-dir", action="append", required=True, dest="text_dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    inv = load_inventory(args.inventory)
    texts = load_texts(args.text_dirs)
    names = sorted(texts)

    # Precompute shingle sets.
    sig = {n: shingles(texts[n]) for n in names}

    # Exact ISBN groups.
    isbn_groups = {}
    for n in names:
        isbn = (inv.get(n, {}) or {}).get("isbn")
        if isbn:
            isbn_groups.setdefault(isbn, []).append(n)
    isbn_dups = {k: v for k, v in isbn_groups.items() if len(v) > 1}

    # Pairwise content similarity.
    pairs = []
    for i in range(len(names)):
        a = names[i]
        sa = sig[a]
        if not sa:
            continue
        for j in range(i + 1, len(names)):
            b = names[j]
            sb = sig[b]
            if not sb:
                continue
            # cheap size pre-filter
            if min(len(sa), len(sb)) / max(len(sa), len(sb)) < 0.3 and \
               not (inv.get(a, {}).get("isbn") and
                    inv.get(a, {}).get("isbn") == inv.get(b, {}).get("isbn")):
                continue
            jac = jaccard(sa, sb)
            same_isbn = (inv.get(a, {}).get("isbn") and
                         inv.get(a, {}).get("isbn") == inv.get(b, {}).get("isbn"))
            if jac >= args.threshold or same_isbn:
                pairs.append((jac, bool(same_isbn), a, b))

    pairs.sort(key=lambda x: (x[1], x[0]), reverse=True)

    # ---- write report ----
    lines = [
        "---",
        "title: Content Duplicate Candidates",
        "type: Report",
        "tags: [duplicates, content-analysis, resources]",
        "---",
        "",
        "# Content Duplicate Candidates",
        "",
        f"> Detected by ISBN match + word-shingle Jaccard ≥ {args.threshold} "
        f"over the first {SAMPLE_CHARS:,} characters.",
        f"> {len(pairs)} candidate pair(s) across {len(names)} documents.",
        "",
    ]

    if isbn_dups:
        lines.append("## Exact ISBN matches (high confidence)\n")
        for isbn, group in sorted(isbn_dups.items()):
            lines.append(f"- **ISBN {isbn}**")
            for n in group:
                lines.append(f"    - {vault_link(inv, n)} `{n}`")
        lines.append("")

    lines.append("## Similar content (by text)\n")
    if not pairs:
        lines.append("_No pairs above threshold._")
    else:
        lines.append("| Similarity | ISBN match | A | B |")
        lines.append("|---:|:---:|---|---|")
        for jac, same_isbn, a, b in pairs:
            la = vault_link(inv, a)
            lb = vault_link(inv, b)
            lines.append(f"| {jac:.0%} | {'✅' if same_isbn else ''} | {la} | {lb} |")
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out}")
    print(f"  {len(isbn_dups)} ISBN-duplicate group(s); {len(pairs)} content pair(s) "
          f"≥ {args.threshold}")
    for jac, same_isbn, a, b in pairs[:12]:
        tag = " [ISBN]" if same_isbn else ""
        print(f"  {jac:.0%}{tag}  {a}  <->  {b}")


if __name__ == "__main__":
    main()
