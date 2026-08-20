"""Config-aware orchestration for the document extraction artifact pipeline."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from extractor.artifacts import (
    ArtifactGeneration,
    begin_generation,
    publish_generation,
    resolve_active,
)
from rag.config import ConfigError, RagConfig, load_config


Command = Tuple[str, ...]
Runner = Callable[[Sequence[str]], None]


@dataclass(frozen=True)
class PipelinePaths:
    books: Path
    resources: Path
    catalog: Path
    output: Path
    notes: Path
    mocs: Path
    vault: Path

    @property
    def text_dirs(self) -> Tuple[Path, Path]:
        return self.books / "text_output", self.resources / "text_output"

    @property
    def inventory(self) -> Path:
        return self.catalog / "resource_inventory.jsonl"

    @property
    def enriched_inventory(self) -> Path:
        return self.catalog / "resource_inventory_enriched.jsonl"

    @property
    def indexed(self) -> Path:
        return self.output / "indexed"

    @classmethod
    def from_config(cls, config: RagConfig) -> "PipelinePaths":
        extractor = config.extractor_model
        if extractor is None:
            raise ConfigError("The active profile has no extractor block")
        required = {
            "books_path": extractor.books_path,
            "resources_path": extractor.resources_path,
            "catalog_path": extractor.catalog_path,
            "output_path": extractor.output_path,
            "obsidian_notes_path": extractor.obsidian_notes_path,
            "mocs_path": extractor.mocs_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ConfigError(f"Extractor path(s) not configured: {', '.join(missing)}")
        return cls(
            books=extractor.books_path,
            resources=extractor.resources_path,
            catalog=extractor.catalog_path,
            output=extractor.output_path,
            notes=extractor.obsidian_notes_path,
            mocs=extractor.mocs_path,
            vault=Path(config["vault_path"]),
        )


@dataclass(frozen=True)
class PipelineStage:
    name: str
    description: str
    commands: Tuple[Command, ...]
    dependencies: Tuple[str, ...] = ()
    inputs: Tuple[Path, ...] = ()
    outputs: Tuple[Path, ...] = ()

    def status(self) -> str:
        if self.inputs and any(not path.exists() for path in self.inputs):
            return "blocked"
        existing = sum(path.exists() for path in self.outputs)
        if self.outputs and existing == len(self.outputs):
            return "complete"
        if existing:
            return "partial"
        return "pending"


DEFAULT_STAGES = (
    "extract",
    "enrich",
    "build-index",
    "build-notes",
    "build-books-index",
    "build-vault-index",
    "build-sqlite",
    "link-mocs",
)


def _module_command(module: str, *args: object) -> Command:
    return (sys.executable, "-m", module, *(str(arg) for arg in args))


def build_pipeline(
    config: RagConfig,
    force: bool = False,
    generation: Optional[ArtifactGeneration] = None,
) -> Dict[str, PipelineStage]:
    """Build the ordered stage graph from one resolved configuration."""
    paths = PipelinePaths.from_config(config)
    if generation is None:
        text_dirs = paths.text_dirs
        enriched_inventory = paths.enriched_inventory
        indexed = paths.indexed
        database = paths.output / "resources.db"
        extraction_logs = (
            paths.books / "extract_progress.log",
            paths.resources / "extract_progress.log",
        )
    else:
        text_dirs = (
            generation.staging / "text" / "books",
            generation.staging / "text" / "resources",
        )
        enriched_inventory = (
            generation.staging / "catalog" / "resource_inventory_enriched.jsonl"
        )
        indexed = generation.staging / "indexed"
        database = generation.staging / "resources.db"
        extraction_logs = (
            generation.staging / "logs" / "books.log",
            generation.staging / "logs" / "resources.log",
        )
    extract_suffix = ("--force",) if force else ()
    extract_commands = []
    for source, output, log_path in zip(
        (paths.books, paths.resources), text_dirs, extraction_logs
    ):
        command = _module_command("extractor.extract_text", source, *extract_suffix)
        if generation is not None:
            command = (*command, "--out", str(output), "--log", str(log_path))
        extract_commands.append(command)

    stages = [
        PipelineStage(
            "analyze",
            "Survey source files and extraction support",
            (
                _module_command("extractor.analyze_files", paths.books),
                _module_command("extractor.analyze_files", paths.resources),
            ),
            outputs=(paths.books / "file_analysis.csv", paths.resources / "file_analysis.csv"),
        ),
        PipelineStage(
            "extract",
            "Extract text from books and resources",
            tuple(extract_commands),
            inputs=(paths.books, paths.resources),
            outputs=(text_dirs[0] / "manifest.json", text_dirs[1] / "manifest.json"),
        ),
        PipelineStage(
            "enrich",
            "Enrich catalog metadata from source files and extracted text",
            (
                _module_command(
                    "extractor.enrich_metadata",
                    "--inventory", paths.inventory,
                    "--source-dir", paths.books,
                    "--source-dir", paths.resources,
                    "--text-dir", text_dirs[0],
                    "--text-dir", text_dirs[1],
                    "--out", enriched_inventory,
                ),
            ),
            dependencies=("extract",),
            inputs=(paths.inventory, *text_dirs),
            outputs=(enriched_inventory,),
        ),
        PipelineStage(
            "build-index",
            "Build enriched per-document JSON and combined JSONL",
            (
                _module_command(
                    "extractor.build_index_documents",
                    "--inventory", enriched_inventory,
                    "--text-dir", text_dirs[0],
                    "--text-dir", text_dirs[1],
                    "--out", indexed,
                ),
            ),
            dependencies=("enrich",),
            inputs=(enriched_inventory, *text_dirs),
            outputs=(indexed / "index_documents.jsonl", indexed / "build_report.json"),
        ),
        PipelineStage(
            "build-notes",
            "Generate Obsidian resource notes",
            (
                _module_command(
                    "extractor.build_obsidian_notes",
                    "--inventory", enriched_inventory,
                    "--text-dir", text_dirs[0],
                    "--text-dir", text_dirs[1],
                    "--generated-dir", paths.mocs,
                    "--out", paths.notes,
                ),
            ),
            dependencies=("enrich",),
            inputs=(enriched_inventory, *text_dirs, paths.mocs),
            outputs=(paths.notes,),
        ),
        PipelineStage(
            "build-books-index",
            "Generate the aggregate Books Index note",
            (
                _module_command(
                    "extractor.build_books_index",
                    "--inventory", enriched_inventory,
                    "--out", paths.mocs / "Books Index.md",
                ),
            ),
            dependencies=("enrich",),
            inputs=(enriched_inventory,),
            outputs=(paths.mocs / "Books Index.md",),
        ),
        PipelineStage(
            "build-vault-index",
            "Build whole-document vault JSONL",
            (
                _module_command(
                    "extractor.build_vault_index",
                    "--vault", paths.vault,
                    "--out", indexed,
                ),
            ),
            inputs=(paths.vault,),
            outputs=(indexed / "vault_documents.jsonl",),
        ),
        PipelineStage(
            "build-sqlite",
            "Build the whole-document SQLite FTS catalog",
            (
                _module_command(
                    "extractor.build_sqlite",
                    "--jsonl", indexed / "index_documents.jsonl",
                    "--jsonl", indexed / "vault_documents.jsonl",
                    "--db", database,
                ),
            ),
            dependencies=("build-index", "build-vault-index"),
            inputs=(indexed / "index_documents.jsonl", indexed / "vault_documents.jsonl"),
            outputs=(database,),
        ),
        PipelineStage(
            "link-mocs",
            "Update generated Topic MOCs with resource-note backlinks",
            (
                _module_command(
                    "extractor.link_mocs",
                    "--inventory", enriched_inventory,
                    "--generated-dir", paths.mocs,
                ),
            ),
            dependencies=("build-notes",),
            inputs=(enriched_inventory, paths.notes, paths.mocs),
            outputs=(paths.mocs,),
        ),
        PipelineStage(
            "dup-detect",
            "Generate the content duplicate-candidate report",
            (
                _module_command(
                    "extractor.dup_detect",
                    "--inventory", enriched_inventory,
                    "--text-dir", text_dirs[0],
                    "--text-dir", text_dirs[1],
                    "--out", paths.mocs / "Content Duplicate Candidates.md",
                ),
            ),
            dependencies=("enrich",),
            inputs=(enriched_inventory, *text_dirs),
            outputs=(paths.mocs / "Content Duplicate Candidates.md",),
        ),
    ]
    return {stage.name: stage for stage in stages}


def resolve_stages(
    pipeline: Mapping[str, PipelineStage],
    selected: Optional[Iterable[str]] = None,
    skipped: Iterable[str] = (),
    include_dependencies: bool = True,
) -> List[PipelineStage]:
    requested = list(selected or DEFAULT_STAGES)
    skipped_set = set(skipped)
    unknown = sorted((set(requested) | skipped_set) - set(pipeline))
    if unknown:
        raise ConfigError(f"Unknown pipeline stage(s): {', '.join(unknown)}")

    wanted = set()

    def add(name: str) -> None:
        if name in skipped_set or name in wanted:
            return
        if include_dependencies:
            for dependency in pipeline[name].dependencies:
                add(dependency)
        wanted.add(name)

    for name in requested:
        add(name)
    return [stage for name, stage in pipeline.items() if name in wanted]


def _subprocess_runner(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def run_pipeline(stages: Sequence[PipelineStage], runner: Runner = _subprocess_runner) -> None:
    for stage in stages:
        print(f"\n==> {stage.name}: {stage.description}")
        for command in stage.commands:
            runner(command)


CORE_STAGES = (
    "extract",
    "enrich",
    "build-index",
    "build-vault-index",
    "build-sqlite",
)
PROJECTION_STAGES = ("build-notes", "build-books-index", "link-mocs")


def run_atomic_pipeline(
    config: RagConfig,
    force: bool = False,
    runner: Runner = _subprocess_runner,
) -> Path:
    """Build core artifacts in isolation and activate them with one pointer swap."""
    paths = PipelinePaths.from_config(config)
    generation = begin_generation(paths.output)
    try:
        staged = build_pipeline(config, force=force, generation=generation)
        core = resolve_stages(
            staged, selected=CORE_STAGES, include_dependencies=False
        )
        run_pipeline(core, runner=runner)
        profile = str(
            config.get("corpus_profile") or config.get("collection_name") or ""
        )
        final = publish_generation(generation, profile=profile)
    except Exception:
        if generation.staging.exists():
            shutil.rmtree(generation.staging)
        raise

    active = resolve_active(paths.output)
    active_generation = ArtifactGeneration(
        root=generation.root,
        generation_id=generation.generation_id,
        staging=active,
        parent_generation_id=generation.parent_generation_id,
    )
    projections = build_pipeline(config, generation=active_generation)
    run_pipeline(
        resolve_stages(
            projections,
            selected=PROJECTION_STAGES,
            include_dependencies=False,
        ),
        runner=runner,
    )
    return final


def print_plan(stages: Sequence[PipelineStage]) -> None:
    for stage in stages:
        print(f"[{stage.status():8s}] {stage.name}: {stage.description}")
        for command in stage.commands:
            print(f"  {shlex.join(command)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-pipeline",
        description="Run the configured extraction artifact pipeline.",
    )
    parser.add_argument("--config", help="Config path (overrides RAG_CONFIG_PATH)")
    parser.add_argument("--stage", action="append", help="Run one stage (repeatable)")
    parser.add_argument("--skip", action="append", default=[], help="Skip one stage")
    parser.add_argument("--no-deps", action="store_true", help="Do not include dependencies")
    parser.add_argument("--dry-run", action="store_true", help="Print status and commands only")
    parser.add_argument("--list-stages", action="store_true", help="List available stages and exit")
    parser.add_argument("--force", action="store_true", help="Force text re-extraction")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        pipeline = build_pipeline(config, force=args.force)
        if args.list_stages:
            print_plan(list(pipeline.values()))
            return
        stages = resolve_stages(
            pipeline,
            selected=args.stage,
            skipped=args.skip,
            include_dependencies=not args.no_deps,
        )
    except ConfigError as exc:
        parser.error(str(exc))

    atomic_full_run = args.stage is None and not args.skip and not args.no_deps
    if args.dry_run:
        if atomic_full_run:
            print("Full runs stage core artifacts in a new generation before activation.")
        print_plan(stages)
    elif atomic_full_run:
        run_atomic_pipeline(config, force=args.force)
    else:
        print(
            "Selected-stage mode is nonpublishing and retains compatibility paths; "
            "run the full pipeline for atomic generation publication."
        )
        run_pipeline(stages)


if __name__ == "__main__":
    main()
