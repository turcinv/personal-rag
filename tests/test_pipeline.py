"""Tests for config-driven extraction pipeline orchestration."""

import subprocess
from pathlib import Path

import pytest
import yaml

from extractor.pipeline import (
    build_pipeline,
    resolve_stages,
    run_atomic_pipeline,
    run_pipeline,
)
from extractor.artifacts import resolve_active
from extractor.build_sqlite import build_database
from rag.config import ConfigError, load_config


def _config(tmp_path: Path, prefix: str = "host"):
    root = tmp_path / prefix
    data = {
        "vault_path": str(root / "vault"),
        "index_path": str(root / "chroma"),
        "store": "chroma",
        "collection_name": "test",
        "embedding_model": "test-model",
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 150,
        "embedding_batch_size": 16,
        "markdown_workers": 1,
        "pdf_workers": 1,
        "extractor": {
            "books_path": str(root / "books"),
            "resources_path": str(root / "resources"),
            "catalog_path": str(root / "catalog"),
            "output_path": str(root / "output"),
            "obsidian_notes_path": str(root / "mocs" / "Resource Notes"),
            "mocs_path": str(root / "mocs"),
        },
    }
    path = tmp_path / f"{prefix}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return load_config(path)


def _modules(stages):
    return [command[2] for stage in stages for command in stage.commands]


def test_default_pipeline_has_safe_dependency_order(tmp_path):
    pipeline = build_pipeline(_config(tmp_path))
    stages = resolve_stages(pipeline)
    names = [stage.name for stage in stages]

    assert names.index("extract") < names.index("enrich") < names.index("build-index")
    assert names.index("build-vault-index") < names.index("build-sqlite")
    assert names.index("build-notes") < names.index("link-mocs")
    assert "analyze" not in names
    assert "dup-detect" not in names


def test_selecting_stage_includes_dependencies(tmp_path):
    stages = resolve_stages(build_pipeline(_config(tmp_path)), selected=["build-sqlite"])
    assert [stage.name for stage in stages] == [
        "extract",
        "enrich",
        "build-index",
        "build-vault-index",
        "build-sqlite",
    ]


def test_no_deps_runs_only_selected_stage(tmp_path):
    stages = resolve_stages(
        build_pipeline(_config(tmp_path)),
        selected=["build-sqlite"],
        include_dependencies=False,
    )
    assert [stage.name for stage in stages] == ["build-sqlite"]


def test_pipeline_commands_use_resolved_config_paths(tmp_path):
    pipeline = build_pipeline(_config(tmp_path))
    enrich = pipeline["enrich"].commands[0]

    assert enrich[1:3] == ("-m", "extractor.enrich_metadata")
    assert str((tmp_path / "host/books").resolve()) in enrich
    assert str((tmp_path / "host/output").resolve()) not in enrich
    assert str((tmp_path / "host/catalog/resource_inventory_enriched.jsonl").resolve()) in enrich


def test_force_is_forwarded_only_to_extract_commands(tmp_path):
    pipeline = build_pipeline(_config(tmp_path), force=True)
    assert all("--force" in command for command in pipeline["extract"].commands)
    assert all("--force" not in command for command in pipeline["enrich"].commands)


def test_local_and_container_profiles_have_same_logical_modules(tmp_path):
    host = resolve_stages(build_pipeline(_config(tmp_path, "host")))
    container = resolve_stages(build_pipeline(_config(tmp_path, "container")))
    assert _modules(host) == _modules(container)


def test_run_pipeline_stops_on_first_failed_command(tmp_path):
    stages = resolve_stages(
        build_pipeline(_config(tmp_path)),
        selected=["extract", "enrich"],
        include_dependencies=False,
    )
    seen = []

    def runner(command):
        seen.append(command)
        if len(seen) == 2:
            raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        run_pipeline(stages, runner=runner)

    assert len(seen) == 2


def test_missing_extractor_block_is_rejected(tmp_path):
    data = {
        "vault_path": str(tmp_path / "vault"),
        "store": "chroma",
        "embedding_batch_size": 16,
        "markdown_workers": 1,
        "pdf_workers": 1,
    }
    path = tmp_path / "markdown-only.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="no extractor block"):
        build_pipeline(load_config(path))


def test_stage_status_reports_blocked_and_complete(tmp_path):
    config = _config(tmp_path)
    pipeline = build_pipeline(config)
    assert pipeline["extract"].status() == "blocked"

    (tmp_path / "host/books/text_output").mkdir(parents=True)
    (tmp_path / "host/resources/text_output").mkdir(parents=True)
    (tmp_path / "host/books/text_output/manifest.json").write_text("[]")
    (tmp_path / "host/resources/text_output/manifest.json").write_text("[]")
    assert pipeline["extract"].status() == "complete"


def _prepare_atomic_inputs(tmp_path):
    for relative in (
        "host/books",
        "host/resources",
        "host/catalog",
        "host/vault/Knowledge",
        "host/mocs",
    ):
        (tmp_path / relative).mkdir(parents=True, exist_ok=True)
    (tmp_path / "host/catalog/resource_inventory.jsonl").write_text(
        "", encoding="utf-8"
    )


def _artifact_runner(command):
    module = command[2]

    def argument(name):
        return Path(command[command.index(name) + 1])

    if module == "extractor.extract_text":
        output = argument("--out")
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text("[]", encoding="utf-8")
    elif module == "extractor.enrich_metadata":
        output = argument("--out")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
    elif module == "extractor.build_index_documents":
        output = argument("--out")
        output.mkdir(parents=True, exist_ok=True)
        (output / "index_documents.jsonl").write_text("", encoding="utf-8")
        (output / "build_report.json").write_text("{}", encoding="utf-8")
    elif module == "extractor.build_vault_index":
        output = argument("--out")
        output.mkdir(parents=True, exist_ok=True)
        (output / "vault_documents.jsonl").write_text("", encoding="utf-8")
    elif module == "extractor.build_sqlite":
        inputs = [
            Path(command[index + 1])
            for index, value in enumerate(command)
            if value == "--jsonl"
        ]
        build_database(inputs, argument("--db"))


def test_atomic_pipeline_routes_core_outputs_to_generation(tmp_path):
    _prepare_atomic_inputs(tmp_path)
    config = _config(tmp_path)

    final = run_atomic_pipeline(config, runner=_artifact_runner)

    assert resolve_active(tmp_path / "host/output") == final
    assert (final / "resources.db").is_file()
    assert (final / "indexed/index_documents.jsonl").is_file()
    assert (tmp_path / "host/output/indexed").resolve() == final / "indexed"
    assert not (tmp_path / "host/books/text_output").exists()
    assert not (tmp_path / "host/resources/extract_progress.log").exists()


def test_failed_atomic_pipeline_preserves_previous_current(tmp_path):
    _prepare_atomic_inputs(tmp_path)
    config = _config(tmp_path)
    first = run_atomic_pipeline(config, runner=_artifact_runner)

    def failing_runner(command):
        if command[2] == "extractor.build_index_documents":
            raise subprocess.CalledProcessError(1, command)
        _artifact_runner(command)

    with pytest.raises(subprocess.CalledProcessError):
        run_atomic_pipeline(config, runner=failing_runner)

    assert resolve_active(tmp_path / "host/output") == first
    generations = tmp_path / "host/output/generations"
    assert not list(generations.glob(".staging-*"))
