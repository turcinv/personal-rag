"""Pre-flight pipeline status check.

Reads config.yaml (same extractor: block used by the pipeline tools) and
reports the state of every extraction step: OK, STALE, or MISSING.

Exit code 0 = all steps OK.  Exit code 1 = any step missing or stale.
Entry point: rag-pipeline-status
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from .utils import load_config

# ── ANSI colours (disabled if stdout is not a tty) ────────────────────────────

def _ansi(code: str, text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[{code}m{text}\033[0m"
    return text

GREEN  = lambda t: _ansi("32", t)
YELLOW = lambda t: _ansi("33", t)
RED    = lambda t: _ansi("31", t)
BOLD   = lambda t: _ansi("1",  t)
DIM    = lambda t: _ansi("2",  t)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ts(path: Path) -> str:
    """Return human-readable mtime for a file/dir, or '—' if absent."""
    if not path.exists():
        return "—"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _count_files(directory: Path, pattern: str = "*") -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


_VAULT_SKIP_DIRS = {".obsidian", ".trash", "Archive", "Templates", "Attachments", ".git"}


def _newest_mtime(directory: Path, pattern: str = "**/*",
                  skip_dirs: set[str] | None = None) -> float:
    """Return the newest mtime among files matching pattern, or 0.

    skip_dirs: if provided, skip any path whose parts include one of these names.
    """
    skip = skip_dirs or set()
    mtimes = [
        p.stat().st_mtime
        for p in directory.glob(pattern)
        if p.is_file() and not (skip and skip.intersection(p.parts))
    ]
    return max(mtimes) if mtimes else 0.0


# ── Status rows ───────────────────────────────────────────────────────────────

def _row(label: str, status: str, detail: str, timestamp: str) -> tuple[str, str, str, str]:
    return label, status, detail, timestamp


def _check_extract(text_dir: Path, source_dir: Path, label: str,
                   build_report: Path) -> tuple[str, str, str, str]:
    extracted = _count_files(text_dir, "*.json")
    if not text_dir.exists():
        # text_output dirs are not synced to Jetson — if build_report.json is present the
        # extraction was already done on the host and only indexed/ was synced over.
        if build_report.exists():
            return _row(label, "OK", "pre-built on host (text_output not synced)", _ts(build_report))
        return _row(label, "MISSING", "no text_output dir", "—")
    ts = _ts(sorted(text_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0]
             if extracted else text_dir)
    if source_dir.exists():
        source_count = (
            sum(1 for _ in source_dir.rglob("*.pdf") if _.is_file())
            + sum(1 for _ in source_dir.rglob("*.epub") if _.is_file())
        )
        detail = f"{extracted} extracted / {source_count} source files"
        # Flag STALE if a source file is newer than the newest extracted JSON
        newest_extracted = _newest_mtime(text_dir, "*.json")
        newest_source = _newest_mtime(source_dir, "**/*.pdf")
        newest_source = max(newest_source, _newest_mtime(source_dir, "**/*.epub"))
        if newest_source > newest_extracted:
            return _row(label, "STALE", detail, ts)
    else:
        detail = f"{extracted} extracted (source dir not found)"
    return _row(label, "OK", detail, ts)


def _check_build_index(indexed_dir: Path) -> tuple[str, str, str, str]:
    report_path = indexed_dir / "build_report.json"
    if not report_path.exists():
        return _row("Enrich / build-index", "MISSING", "no build_report.json", "—")
    try:
        report = json.loads(report_path.read_text())
        matched   = report.get("matched", "?")
        inventory = report.get("inventory_records", "?")
        unmatched = report.get("in_inventory_without_text", [])
        detail = f"{matched}/{inventory} matched"
        if unmatched:
            detail += f"  ({len(unmatched)} without text)"
    except Exception:
        detail = "build_report.json unreadable"
    return _row("Enrich / build-index", "OK", detail, _ts(report_path))


def _check_build_notes(notes_dir: Path, inventory_path: Path) -> tuple[str, str, str, str]:
    if not notes_dir.exists():
        return _row("Build notes", "MISSING", "Resource Notes dir absent", "—")
    note_count = _count_files(notes_dir, "*.md")
    if inventory_path.exists():
        inv_count = sum(1 for _ in inventory_path.read_text().splitlines() if _.strip())
        detail = f"{note_count} notes / {inv_count} inventory records"
    else:
        detail = f"{note_count} notes"
    newest = sorted(notes_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    ts = _ts(newest[0]) if newest else "—"
    return _row("Build notes", "OK", detail, ts)


def _check_sqlite(db_path: Path, indexed_dir: Path) -> tuple[str, str, str, str]:
    if not db_path.exists():
        # On the Jetson, resources.db is not synced but can be rebuilt from indexed/ JSONL.
        jsonl = indexed_dir / "index_documents.jsonl"
        if jsonl.exists():
            return _row("Build SQLite", "REBUILD", "run jetson-build-sqlite to generate", "—")
        return _row("Build SQLite", "MISSING", "resources.db absent", "—")
    size_mb = db_path.stat().st_size / (1024 * 1024)
    detail = f"{size_mb:.0f} MB"
    return _row("Build SQLite", "OK", detail, _ts(db_path))


def _check_vault_jsonl(indexed_dir: Path, vault_path: Path) -> tuple[str, str, str, str]:
    jsonl_path = indexed_dir / "vault_documents.jsonl"
    if not jsonl_path.exists():
        return _row("Vault JSONL", "MISSING", "vault_documents.jsonl absent", "—")
    line_count = sum(1 for _ in jsonl_path.open())
    detail = f"{line_count} docs"
    if vault_path.exists():
        newest_vault = _newest_mtime(vault_path, "**/*.md", skip_dirs=_VAULT_SKIP_DIRS)
        if newest_vault > jsonl_path.stat().st_mtime:
            return _row("Vault JSONL", "STALE", detail + "  (vault has newer notes)", _ts(jsonl_path))
    return _row("Vault JSONL", "OK", detail, _ts(jsonl_path))


# ── Formatting ────────────────────────────────────────────────────────────────

_STATUS_COLOR = {"OK": GREEN, "STALE": YELLOW, "REBUILD": YELLOW, "MISSING": RED}
_COL_W = (22, 7, 38, 18)  # label, status, detail, timestamp


def _fmt_row(label: str, status: str, detail: str, ts: str) -> str:
    color = _STATUS_COLOR.get(status, lambda t: t)
    return (
        f"  {label:<{_COL_W[0]}}"
        f"{color(f'{status:<{_COL_W[1]}}')}"
        f"  {detail:<{_COL_W[2]}}"
        f"  {DIM(ts)}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    config = load_config()
    ext = config.get("extractor", {})

    output_path   = Path(ext.get("output_path", "~/Documents/knowledge-base-index")).expanduser()
    books_path    = Path(ext.get("books_path", "")).expanduser()
    resources_path = Path(ext.get("resources_path", "")).expanduser()
    catalog_path  = Path(ext.get("catalog_path", "")).expanduser()
    notes_path    = Path(ext.get("obsidian_notes_path", "")).expanduser()
    vault_path    = Path(config.get("vault_path", "")).expanduser()

    indexed_dir         = output_path / "indexed"
    text_books_dir      = output_path / "text_output_books"
    text_resources_dir  = output_path / "text_output_resources"
    db_path             = output_path / "resources.db"
    inventory_enriched  = catalog_path / "resource_inventory_enriched.jsonl"

    sep = "─" * 90
    print()
    print(BOLD("Pipeline status"))
    print(sep)

    build_report = indexed_dir / "build_report.json"
    rows = [
        _check_extract(text_books_dir, books_path, "Extract books", build_report),
        _check_extract(text_resources_dir, resources_path, "Extract resources", build_report),
        _check_build_index(indexed_dir),
        _check_build_notes(notes_path, inventory_enriched),
        _check_sqlite(db_path, indexed_dir),
        _check_vault_jsonl(indexed_dir, vault_path),
    ]

    any_bad = False
    for label, status, detail, ts in rows:
        print(_fmt_row(label, status, detail, ts))
        if status in ("MISSING", "STALE"):
            any_bad = True

    print(sep)
    has_rebuild = any(status == "REBUILD" for _, status, _, _ in rows)
    if any_bad:
        print(RED("  One or more steps are MISSING or STALE. Run the pipeline before indexing."))
        print()
        sys.exit(1)
    elif has_rebuild:
        print(YELLOW("  Optional steps need rebuilding (see REBUILD rows above)."))
        print(GREEN("  Core steps OK. Ready to index."))
        print()
        sys.exit(0)
    else:
        print(GREEN("  All steps OK. Ready to index."))
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
