"""Offline tests for generation-aware artifact sync planning.

No network, no ssh, no rsync: only the pure command planner is exercised, over a
fixture generation published under tmp_path.
"""

import json

import pytest

from extractor.artifacts import begin_generation, publish_generation
from extractor.build_sqlite import build_database
from extractor.sync import plan_generation_sync, run_generation_sync
from rag.config import ConfigError


def _publish(root, indexed_documents=None):
    generation = begin_generation(root)
    indexed_documents = indexed_documents or {}
    for relative in (
        "text/books/manifest.json",
        "text/resources/manifest.json",
        "indexed/build_report.json",
    ):
        path = generation.staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]" if path.name == "manifest.json" else "{}", encoding="utf-8")
    (generation.staging / "catalog/resource_inventory_enriched.jsonl").parent.mkdir(
        parents=True, exist_ok=True
    )
    (generation.staging / "catalog/resource_inventory_enriched.jsonl").write_text(
        "", encoding="utf-8"
    )
    index_jsonl = generation.staging / "indexed/index_documents.jsonl"
    vault_jsonl = generation.staging / "indexed/vault_documents.jsonl"
    index_jsonl.write_text("", encoding="utf-8")
    vault_jsonl.write_text("", encoding="utf-8")
    for name, payload in indexed_documents.items():
        (generation.staging / "indexed" / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    build_database([index_jsonl, vault_jsonl], generation.staging / "resources.db")
    return publish_generation(generation)


def test_plan_transfers_generation_then_aliases_then_current(tmp_path):
    final = _publish(tmp_path)
    plan = plan_generation_sync(tmp_path, "jetson", "~/knowledge-base-index")

    # Generation contents go first; the current-pointer activation goes last.
    assert plan[0][:2] == ("rsync", "-a")
    assert plan[0][-2].endswith(f"generations/{final.name}/")
    assert plan[-1][-1] == "jetson:~/knowledge-base-index/"
    assert plan[-1][-2].endswith("/current")

    middle_targets = " ".join(part for command in plan[1:-1] for part in command)
    for alias in ("indexed", "resources.db", "text_output_books", "text_output_resources"):
        assert alias in middle_targets


def test_plan_never_includes_lexical_or_chroma(tmp_path):
    _publish(tmp_path)
    (tmp_path / "lexical_index").mkdir()
    (tmp_path / "lexical_index/test.db").write_text("x", encoding="utf-8")
    (tmp_path / "chroma_db").mkdir()

    joined = " ".join(part for command in plan_generation_sync(tmp_path, "h", "/r") for part in command)
    assert "lexical_index" not in joined
    assert "chroma_db" not in joined


def test_plan_requires_active_generation(tmp_path):
    from extractor.artifacts import ArtifactError

    with pytest.raises(ArtifactError, match="No active artifact generation"):
        plan_generation_sync(tmp_path, "jetson", "/remote")


def test_plan_requires_host_and_remote(tmp_path):
    _publish(tmp_path)
    with pytest.raises(ConfigError, match="remote host"):
        plan_generation_sync(tmp_path, "", "/remote")
    with pytest.raises(ConfigError, match="remote output path"):
        plan_generation_sync(tmp_path, "jetson", "")


def test_run_executes_planned_commands_in_order(tmp_path):
    _publish(tmp_path)
    seen = []
    run_generation_sync(tmp_path, "jetson", "/remote", runner=seen.append)

    assert seen[0][-2].endswith("/")
    assert seen[-1][-2].endswith("/current")
    assert all(command[0] == "rsync" for command in seen)
