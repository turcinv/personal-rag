import json
import os
from pathlib import Path

import pytest

from extractor import artifacts
from extractor.artifacts import (
    ArtifactError,
    begin_generation,
    build_manifest,
    publish_generation,
    resolve_active,
    validate_generation,
    write_manifest,
)
from extractor.build_sqlite import build_database


def _complete_generation(root: Path, indexed_documents=None):
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

    catalog = generation.staging / "catalog/resource_inventory_enriched.jsonl"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("", encoding="utf-8")
    index_jsonl = generation.staging / "indexed/index_documents.jsonl"
    vault_jsonl = generation.staging / "indexed/vault_documents.jsonl"
    index_jsonl.write_text("", encoding="utf-8")
    vault_jsonl.write_text("", encoding="utf-8")
    for name, payload in indexed_documents.items():
        (generation.staging / "indexed" / name).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    build_database(
        [index_jsonl, vault_jsonl], generation.staging / "resources.db"
    )
    return generation


def test_publish_generation_validates_and_switches_all_aliases(tmp_path):
    generation = _complete_generation(
        tmp_path,
        {"book.json": {"id": "book", "file_name": "book.pdf", "text": "x"}},
    )

    final = publish_generation(generation, profile="test")

    assert resolve_active(tmp_path) == final
    payload = validate_generation(final, expected_generation_id=final.name)
    assert payload["profile"] == "test"
    assert payload["counts"]["indexed_document"] == 1
    assert (tmp_path / "indexed").resolve() == final / "indexed"
    assert (tmp_path / "resources.db").resolve() == final / "resources.db"
    assert (tmp_path / "text_output_books").resolve() == final / "text/books"
    assert (tmp_path / "text_output_resources").resolve() == final / "text/resources"


def test_incomplete_or_malformed_generation_cannot_publish(tmp_path):
    generation = begin_generation(tmp_path)
    (generation.staging / "broken.json").write_text("{", encoding="utf-8")

    with pytest.raises(ArtifactError, match="Invalid artifact"):
        build_manifest(generation)
    assert not (tmp_path / "current").exists()


def test_checksum_tampering_is_detected(tmp_path):
    generation = _complete_generation(tmp_path)
    write_manifest(generation)
    target = generation.staging / "indexed/build_report.json"
    target.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ArtifactError, match="size mismatch|checksum mismatch"):
        validate_generation(generation.staging)


def test_alias_conflict_is_rejected_before_current_switch(tmp_path):
    first = _complete_generation(tmp_path)
    first_final = publish_generation(first)
    second = _complete_generation(tmp_path)
    alias = tmp_path / "indexed"
    alias.unlink()
    alias.mkdir()

    with pytest.raises(ArtifactError, match="legacy path"):
        publish_generation(second)

    assert resolve_active(tmp_path) == first_final
    assert second.staging.exists()


def test_interrupted_current_swap_keeps_previous_generation_active(
    tmp_path, monkeypatch
):
    first = _complete_generation(tmp_path)
    first_final = publish_generation(first)
    second = _complete_generation(tmp_path)
    real_replace = os.replace

    def fail_current_swap(source, destination):
        if Path(destination) == tmp_path / "current":
            raise OSError("simulated disk failure")
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_current_swap)
    with pytest.raises(OSError, match="simulated disk failure"):
        publish_generation(second)

    assert resolve_active(tmp_path) == first_final
    assert second.final.exists()


def test_new_generation_has_no_stale_documents_and_reports_removals(tmp_path):
    first = _complete_generation(
        tmp_path,
        {"old.json": {"id": "old", "file_name": "old.pdf", "text": "old"}},
    )
    first_final = publish_generation(first)
    second = _complete_generation(
        tmp_path,
        {"new.json": {"id": "new", "file_name": "new.pdf", "text": "new"}},
    )
    second_final = publish_generation(second)

    payload = validate_generation(second_final)
    assert "indexed/old.json" in payload["removed_since_parent"]
    assert not (second_final / "indexed/old.json").exists()
    assert (first_final / "indexed/old.json").exists()


def test_sqlite_sidecars_are_never_manifested(tmp_path):
    generation = _complete_generation(tmp_path)
    (generation.staging / "resources.db-wal").write_bytes(b"stale")

    with pytest.raises(ArtifactError, match="sidecar"):
        build_manifest(generation)
