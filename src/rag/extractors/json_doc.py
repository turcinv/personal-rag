"""Pre-extracted document JSON extractor (doc-text-extractor `indexed/*.json`).

Each file carries the full extracted `text` plus enriched metadata (title,
primary_topic, resource_type, tags, confidence). Because the text is
pre-extracted and stable, the content-hashed IDs are deterministic and re-runs
are idempotent — unlike live PDF parsing."""

import json
from pathlib import Path

from ..chunking import chunk_paragraphs, split_by_headings, stable_id


def extract_json_doc(json_path: Path, max_chars: int, overlap: int):
    """Index one pre-extracted document JSON → (ids, documents, metadatas, error)."""
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [], [], f"skip json read error: {json_path.name}: {exc}"

    text = (obj.get("text") or "").strip()
    if len(text) < 40:
        return [], [], [], None  # empty / failed extraction — nothing to index

    file_name = str(obj.get("file_name") or json_path.stem)
    tags_value = obj.get("tags") or []
    tags = ", ".join(str(t) for t in tags_value) if isinstance(tags_value, list) else str(tags_value)
    meta_base = {
        "path": file_name,
        "title": str(obj.get("title") or file_name),
        "type": str(obj.get("resource_type") or obj.get("source_group") or "resource"),
        "domain": str(obj.get("primary_topic") or ""),
        "status": "",
        "source": "pdf",  # keep books/resources under the existing `--source pdf` filter
        "confidence": str(obj.get("confidence") or ""),
        "tags": tags,
        "wikilinks": "",
    }

    # Structure-aware chunking. Real document structure only shows up as ##+
    # Markdown headings in the pre-extracted text (a lone # is almost always a
    # code comment in book PDFs, e.g. "# load the data" — see split_by_headings
    # min_level). When ##+ headings exist, split into sections and carry the real
    # heading; otherwise treat the whole doc as one section. Either way, chunk on
    # paragraph/sentence boundaries so books are never cut mid-section/mid-word.
    sections = split_by_headings(text, min_level=2)
    structured = bool(sections) and not (len(sections) == 1 and sections[0][0] == "Document")
    if not structured:
        sections = [("", text)]

    ids, documents, metadatas = [], [], []
    for section_index, (heading, body) in enumerate(sections):
        # "Document" is the sentinel for text before the first heading — not a
        # real heading, so store it as empty.
        section_heading = "" if heading == "Document" else heading
        for chunk_index, chunk in enumerate(chunk_paragraphs(body, max_chars, overlap)):
            ids.append(stable_id(file_name, section_index, chunk_index, chunk))
            documents.append(chunk)
            metadatas.append({**meta_base, "heading": section_heading})
    return ids, documents, metadatas, None
