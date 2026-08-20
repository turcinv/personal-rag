"""Tests for typed configuration, overlays, path resolution, and preflight."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from rag.config import ConfigError, RagConfig, load_config, preflight_sources


_ENV_OVERRIDES = (
    "RAG_CONFIG_PATH",
    "RAG_VAULT_PATH",
    "RAG_PDF_BOOKS_PATH",
    "RAG_PDF_RESOURCES_PATH",
    "RAG_JSON_PATH",
    "RAG_INDEX_PATH",
    "RAG_BOOKS_PATH",
    "RAG_RESOURCES_PATH",
    "RAG_CATALOG_PATH",
    "RAG_OUTPUT_PATH",
    "RAG_OBSIDIAN_NOTES_PATH",
    "RAG_MOCS_PATH",
)


@pytest.fixture(autouse=True)
def clear_config_environment(monkeypatch):
    for name in _ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _minimal(vault_path="vault"):
    return {
        "vault_path": vault_path,
        "index_path": "data/chroma",
        "store": "chroma",
        "collection_name": "test_collection",
        "embedding_model": "test-model",
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 150,
        "embedding_batch_size": 16,
        "markdown_workers": 1,
        "pdf_workers": 1,
    }


def test_load_config_returns_typed_mapping_and_resolves_relative_paths(tmp_path):
    config_path = _write(tmp_path / "profile.yaml", _minimal())

    config = load_config(config_path)

    assert isinstance(config, RagConfig)
    assert config.config_path == config_path.resolve()
    assert config["vault_path"] == str((tmp_path / "vault").resolve())
    assert config["index_path"] == str((tmp_path / "data/chroma").resolve())
    assert config.api_model.port == 8000


def test_overlay_deep_merges_and_uses_selected_profile_directory(tmp_path):
    base = _minimal()
    base["api"] = {"host": "127.0.0.1", "port": 8000}
    _write(tmp_path / "base.yaml", base)
    _write(
        tmp_path / "personal.yaml",
        {"extends": "base.yaml", "api": {"port": 9000}, "collection_name": "overlay"},
    )

    config = load_config(tmp_path / "personal.yaml")

    assert config["collection_name"] == "overlay"
    assert config["api"] == {"host": "127.0.0.1", "port": 9000}
    assert config.api_model.host == "127.0.0.1"
    assert config.api_model.port == 9000
    assert config["vault_path"] == str((tmp_path / "vault").resolve())


def test_environment_overrides_take_precedence_and_are_resolved(tmp_path, monkeypatch):
    data = _minimal()
    data["pdf_sources"] = [
        {"path": "books", "type": "book"},
        {"path": "resources", "type": "resource"},
    ]
    data["json_sources"] = [
        {"id": "json:catalog", "path": "indexed", "type": "catalog"}
    ]
    config_path = _write(tmp_path / "profile.yaml", data)
    monkeypatch.setenv("RAG_VAULT_PATH", "env-vault")
    monkeypatch.setenv("RAG_PDF_BOOKS_PATH", "env-books")
    monkeypatch.setenv("RAG_JSON_PATH", "env-json")

    config = load_config(config_path)

    assert config["vault_path"] == str((tmp_path / "env-vault").resolve())
    assert config["pdf_sources"][0]["path"] == str((tmp_path / "env-books").resolve())
    assert config["json_sources"] == [
        {
            "id": "json:catalog",
            "path": str((tmp_path / "env-json").resolve()),
            "type": "catalog",
        }
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg.update({"unknown_option": True}), "Unknown top-level"),
        (lambda cfg: cfg.update({"embedding_batch_size": 0}), "embedding_batch_size"),
        (lambda cfg: cfg.update({"chunk_overlap_chars": 1200}), "chunk_overlap_chars"),
        (lambda cfg: cfg.update({"hybrid_weights": [1.0]}), "hybrid_weights"),
        (lambda cfg: cfg.update({"api": {"port": 70000}}), "api.port"),
    ],
)
def test_invalid_configuration_is_rejected(tmp_path, mutation, message):
    data = _minimal()
    mutation(data)
    config_path = _write(tmp_path / "invalid.yaml", data)

    with pytest.raises(ConfigError, match=message):
        load_config(config_path)


def test_unknown_nested_keys_are_rejected(tmp_path):
    data = _minimal()
    data["pdf_sources"] = [{"path": "books", "typo": "book"}]

    with pytest.raises(ConfigError, match=r"pdf_sources\[0\]"):
        load_config(_write(tmp_path / "invalid.yaml", data))


def test_preflight_reports_available_missing_and_disabled_sources(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    data = _minimal(str(vault))
    data["pdf_sources"] = [{"id": "pdf:books", "path": "missing-books", "type": "book"}]
    config = load_config(_write(tmp_path / "profile.yaml", data))

    checks = preflight_sources(config)
    by_id = {check.source_id: check for check in checks}

    assert by_id["markdown:vault"].state == "available"
    assert by_id["pdf:books"].state == "missing"
    assert by_id["json:disabled"].state == "disabled"
    assert by_id["extractor:disabled"].state == "disabled"


def test_config_doctor_json_and_strict_exit(tmp_path, monkeypatch, capsys):
    from rag.config_cli import main

    config_path = _write(tmp_path / "profile.yaml", _minimal("missing-vault"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["rag-config", "--config", str(config_path), "--json", "--strict"],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["config_path"] == str(config_path.resolve())
    assert output["sources"][0]["state"] == "missing"


def test_personal_overlay_matches_default_profile(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    default = load_config(root / "config.yaml")
    personal = load_config(root / "config.personal.yaml")

    assert dict(personal) == dict(default)
    assert personal.config_path.name == "config.personal.yaml"


def test_duplicate_effective_source_ids_are_rejected(tmp_path):
    data = _minimal()
    data["pdf_sources"] = [
        {"path": "books-a", "type": "book"},
        {"path": "books-b", "type": "book"},
    ]

    with pytest.raises(ConfigError, match="Duplicate source id 'pdf:book'"):
        load_config(_write(tmp_path / "duplicate.yaml", data))


def test_source_id_cannot_collide_with_reserved_vault_id(tmp_path):
    data = _minimal()
    data["json_sources"] = [{"id": "markdown:vault", "path": "indexed"}]

    with pytest.raises(ConfigError, match="Duplicate source id 'markdown:vault'"):
        load_config(_write(tmp_path / "duplicate.yaml", data))


def test_json_environment_override_rejects_multiple_sources(tmp_path, monkeypatch):
    data = _minimal()
    data["json_sources"] = [
        {"id": "json:a", "path": "indexed-a"},
        {"id": "json:b", "path": "indexed-b"},
    ]
    monkeypatch.setenv("RAG_JSON_PATH", "override")

    with pytest.raises(ConfigError, match="RAG_JSON_PATH is ambiguous"):
        load_config(_write(tmp_path / "ambiguous.yaml", data))
