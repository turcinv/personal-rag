"""Source registry for the shared incremental indexing engine."""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set

from extractor.artifacts import validate_generation

from .json_doc import extract_json_doc, load_indexable_json
from .markdown import extract_md_file, should_exclude
from .pdf import extract_pdf_file

__all__ = [
    "Source",
    "iter_sources",
    "extract_md_file",
    "extract_pdf_file",
    "extract_json_doc",
    "should_exclude",
]


@dataclass(frozen=True)
class Source:
    """One independently reconcilable indexing source."""

    source_id: str
    kind: str
    label: str
    root: Path
    files: Sequence[Path]
    workers: int
    extract: Optional[Callable]
    file_key: Callable[[Path], str]
    state: str = "available"
    detail: str = ""
    excluded_files: int = 0

    @property
    def prune_candidate(self) -> bool:
        return self.state == "available" and bool(self.files or self.excluded_files)


def _source_id(kind: str, source: dict, index: int) -> str:
    return source.get("id") or f"{kind}:{source.get('type') or index}"


def _path_state(path: Path, enabled: bool) -> tuple:
    if not enabled:
        return "disabled", "disabled by config"
    if not path.exists():
        return "missing", "path does not exist"
    if not path.is_dir():
        return "unreadable", "path is not a directory"
    if not os.access(path, os.R_OK):
        return "unreadable", "path is not readable"
    return "available", ""


def _dir_source(kind, source, index, glob, workers, make_extract, make_file_key):
    root = Path(source["path"]).expanduser().resolve()
    source_id = _source_id(kind, source, index)
    state, detail = _path_state(root, bool(source.get("enabled", True)))
    files = sorted(root.glob(glob)) if state == "available" else []
    label = f"{kind.upper()} source [{source.get('type')}]" if source.get("type") else f"{kind.upper()} source"
    if state != "available":
        label = f"{label}: {state} — {root}"
    else:
        label = f"{label}: {len(files)} files — {root.name}"
    return Source(
        source_id=source_id,
        kind=kind,
        label=label,
        root=root,
        files=files,
        workers=workers,
        extract=make_extract(source) if state == "available" else None,
        file_key=make_file_key,
        state=state,
        detail=detail,
    )


def _normalize_source_group(value: object) -> Optional[str]:
    if not value:
        return None
    group = str(value).strip().casefold()
    return {"books": "book", "resources": "resource"}.get(group, group)


def _json_files(root: Path) -> List[Path]:
    """Snapshot a legacy JSON directory or a validated managed generation."""
    root = root.resolve()
    generation = root.parent
    if root.name == "indexed" and generation.parent.name == "generations":
        manifest = validate_generation(
            generation, expected_generation_id=generation.name
        )
        allowed = [
            generation / item["path"]
            for item in manifest["artifacts"]
            if item.get("role") == "indexed_document"
        ]
        return sorted(path for path in allowed if path.parent == root)
    return sorted(root.glob("*.json"))


def _json_covered_filenames(
    config: dict, resolved_sources: Optional[Sequence[tuple]] = None
) -> Dict[Optional[str], Set[str]]:
    """Return indexable JSON filenames grouped by their originating PDF source."""
    covered: Dict[Optional[str], Set[str]] = {}
    if resolved_sources is None:
        resolved_sources = []
        for index, source in enumerate(config.get("json_sources", [])):
            root = Path(source["path"]).expanduser().resolve()
            state, detail = _path_state(root, bool(source.get("enabled", True)))
            files = _json_files(root) if state == "available" else []
            resolved_sources.append((index, source, root, files, state, detail))
    for _, source, _, files, state, _ in resolved_sources:
        if state != "available":
            continue
        for path in files:
            obj, error = load_indexable_json(path)
            if error or obj is None:
                continue
            name = obj.get("file_name")
            if not isinstance(name, str) or not name.strip():
                continue
            group = _normalize_source_group(obj.get("source_group"))
            covered.setdefault(group, set()).add(name.strip())
    return covered


def iter_sources(config: dict, vault_path: Path, max_chars: int, overlap: int):
    """Yield source-scoped Markdown, PDF, and enriched-JSON definitions."""
    md_workers = int(config.get("markdown_workers", 1))
    pdf_workers = int(config.get("pdf_workers", 1))

    vault_state, vault_detail = _path_state(vault_path, True)
    md_files = []
    if vault_state == "available":
        md_files = [
            path
            for path in sorted(vault_path.rglob("*.md"))
            if not should_exclude(path, vault_path, config)
        ]
    yield Source(
        source_id="markdown:vault",
        kind="markdown",
        label=(
            f"Markdown: {len(md_files)} files"
            if vault_state == "available"
            else f"Markdown: {vault_state} — {vault_path}"
        ),
        root=vault_path,
        files=md_files,
        workers=md_workers,
        extract=(
            lambda path: extract_md_file(path, vault_path, config, max_chars, overlap)
            if vault_state == "available"
            else None
        ),
        file_key=lambda path: path.relative_to(vault_path).as_posix(),
        state=vault_state,
        detail=vault_detail,
    )

    resolved_json_sources = []
    for index, source_config in enumerate(config.get("json_sources", [])):
        root = Path(source_config["path"]).expanduser().resolve()
        state, detail = _path_state(
            root, bool(source_config.get("enabled", True))
        )
        files = _json_files(root) if state == "available" else []
        resolved_json_sources.append(
            (index, source_config, root, files, state, detail)
        )

    json_covered = _json_covered_filenames(config, resolved_json_sources)
    pdf_source_configs = config.get("pdf_sources", [])

    for index, source_config in enumerate(pdf_source_configs):
        source = _dir_source(
            "pdf",
            source_config,
            index,
            "*.pdf",
            pdf_workers,
            lambda item: (
                lambda path, type_=item.get("type", "resource"): extract_pdf_file(
                    path, type_, max_chars, overlap
                )
            ),
            lambda path: path.name,
        )
        if json_covered and source.files:
            group = _normalize_source_group(source_config.get("type"))
            covered_names = set(json_covered.get(group, set()))
            if len(pdf_source_configs) == 1:
                covered_names.update(json_covered.get(None, set()))
            kept = [path for path in source.files if path.name not in covered_names]
            skipped = len(source.files) - len(kept)
            label = source.label + (f" ({skipped} covered by JSON, skipped)" if skipped else "")
            source = replace(
                source,
                files=kept,
                label=label,
                excluded_files=skipped,
            )
        yield source

    for index, source_config, root, files, state, detail in resolved_json_sources:
        source_id = _source_id("json", source_config, index)
        label = (
            f"JSON source [{source_config.get('type')}]"
            if source_config.get("type")
            else "JSON source"
        )
        if state == "available":
            label = f"{label}: {len(files)} files — {root.name}"
        else:
            label = f"{label}: {state} — {root}"
        yield Source(
            source_id=source_id,
            kind="json",
            label=label,
            root=root,
            files=files,
            workers=pdf_workers,
            extract=(
                lambda path: extract_json_doc(path, max_chars, overlap)
                if state == "available"
                else None
            ),
            file_key=lambda path: path.name,
            state=state,
            detail=detail,
        )
