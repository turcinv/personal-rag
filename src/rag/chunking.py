"""Pure text helpers shared by every extractor: heading split, char chunking,
wikilink extraction, and content-hashed chunk IDs. No I/O, no embedding."""

import hashlib
import re


def extract_wikilinks(text: str):
    return sorted(set(re.findall(r"\[\[([^\]|#]+)", text)))


# Vault convention: every note ends with navigation-only sections ("# Related
# Topics" and "## Potential New Notes") that are pure wikilink lists. They carry
# graph structure (captured separately via extract_wikilinks) but no semantic
# content — embedding them dilutes retrieval with title soup (~16% of corpus).
_NAV_TAIL = re.compile(r"(?m)^#{1,6}\s*(Related Topics|Potential New Notes)\s*$")


def strip_navigation_tail(text: str) -> str:
    """Drop everything from the first navigation heading to the end of the note."""
    m = _NAV_TAIL.search(text)
    return text[: m.start()].rstrip() if m else text


def strip_wikilink_syntax(text: str) -> str:
    """Inline [[target|alias]] -> alias, [[target]] -> target.

    Embedding models see plain words instead of bracket noise; the link graph
    itself is preserved in chunk metadata by extract_wikilinks (run it first).
    """
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    return re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)


def split_by_headings(text: str, min_level: int = 1):
    """Split text into (heading, body) sections at Markdown headings.

    ``min_level`` is the minimum heading depth that counts as a section break.
    The default (1) treats any line starting with ``#`` as a heading — correct
    for hand-written vault notes. Pass ``min_level=2`` for pre-extracted book/
    resource text, where a lone ``#`` is almost always a code comment (e.g.
    ``# load the data``) rather than a real heading, and only ``##``+ lines mark
    genuine document structure. Sections with an empty body are dropped."""
    sections = []
    current_heading = "Document"
    current_lines = []
    for line in text.splitlines():
        if min_level <= 1:
            is_heading = line.startswith("#")
        else:
            is_heading = bool(re.match(r"#{%d,}\s" % min_level, line))
        if is_heading:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
                current_lines = []
            current_heading = line.strip("#").strip() or "Document"
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, t) for h, t in sections if t.strip()]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_paragraphs(text: str, max_chars: int, overlap: int):
    """Chunk text on natural boundaries, packing whole paragraphs up to
    ``max_chars``. A paragraph longer than ``max_chars`` is split on sentence
    boundaries and its sentences packed; a single sentence longer than
    ``max_chars`` falls back to the fixed char-window chunker (``overlap`` is
    applied only in that last-resort case). Never splits mid-sentence otherwise —
    the fix for flat char-window chunking cutting books mid-section/mid-word."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    buf = ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            sent_buf = ""
            for sent in _SENTENCE_BOUNDARY.split(para):
                sent = sent.strip()
                if not sent:
                    continue
                if len(sent) > max_chars:
                    if sent_buf:
                        chunks.append(sent_buf)
                        sent_buf = ""
                    chunks.extend(chunk_text(sent, max_chars, overlap))  # monster sentence
                elif sent_buf and len(sent_buf) + 1 + len(sent) > max_chars:
                    chunks.append(sent_buf)
                    sent_buf = sent
                else:
                    sent_buf = f"{sent_buf} {sent}" if sent_buf else sent
            if sent_buf:
                chunks.append(sent_buf)
        elif buf and len(buf) + 2 + len(para) > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_text(text: str, max_chars: int, overlap: int):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(*parts):
    """Deterministic chunk ID. Hashing the full chunk text makes identical
    content yield the same ID (the basis for incremental/idempotent indexing)."""
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
