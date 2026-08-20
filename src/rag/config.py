"""Typed configuration loading, validation, overlays, and source preflight."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_CONFIG_NAME = "config.yaml"

_TOP_LEVEL_KEYS = {
    "api",
    "chunk_max_chars",
    "chunk_overlap_chars",
    "collection_name",
    "corpus_profile",
    "embedding_batch_size",
    "embedding_dimension",
    "embedding_model",
    "embedding_revision",
    "exclude_dirs",
    "exclude_filename_patterns",
    "exclude_files",
    "extractor",
    "generation",
    "hybrid_fetch_k",
    "hybrid_rrf_k",
    "hybrid_weights",
    "index_path",
    "json_sources",
    "lexical_path",
    "log_db_path",
    "log_path",
    "log_queries",
    "log_retention_days",
    "manifest_path",
    "markdown_workers",
    "offline",
    "pdf_sources",
    "pdf_workers",
    "prune_max_chunks",
    "prune_max_fraction",
    "query_instruction",
    "rerank_default",
    "rerank_fetch_k",
    "reranker_model",
    "store",
    "tag_fetch_k",
    "vault_path",
}
_SOURCE_KEYS = {"enabled", "id", "path", "type"}
_EXTRACTOR_KEYS = {
    "books_path",
    "catalog_path",
    "mocs_path",
    "obsidian_notes_path",
    "output_path",
    "resources_path",
}
_GENERATION_KEYS = {
    "api_key_env",
    "base_url",
    "max_tokens",
    "model",
    "provider",
    "temperature",
    "timeout",
}
_API_KEYS = {
    "host",
    "inference_concurrency",
    "jwt_audience",
    "jwt_issuer",
    "port",
}
_PATH_KEYS = {
    "vault_path",
    "index_path",
    "log_path",
    "log_db_path",
    "lexical_path",
    "manifest_path",
}
_EXTRACTOR_ENV = {
    "RAG_BOOKS_PATH": "books_path",
    "RAG_RESOURCES_PATH": "resources_path",
    "RAG_CATALOG_PATH": "catalog_path",
    "RAG_OUTPUT_PATH": "output_path",
    "RAG_OBSIDIAN_NOTES_PATH": "obsidian_notes_path",
    "RAG_MOCS_PATH": "mocs_path",
}


class ConfigError(ValueError):
    """Raised when configuration is malformed or unsafe to interpret."""


@dataclass(frozen=True)
class SourceConfig:
    source_id: str
    path: Path
    kind: str
    type: Optional[str] = None
    enabled: bool = True


@dataclass(frozen=True)
class ExtractorConfig:
    books_path: Optional[Path] = None
    resources_path: Optional[Path] = None
    catalog_path: Optional[Path] = None
    output_path: Optional[Path] = None
    obsidian_notes_path: Optional[Path] = None
    mocs_path: Optional[Path] = None


@dataclass(frozen=True)
class GenerationConfig:
    provider: str
    model: str
    max_tokens: int = 1024
    temperature: float = 0.0
    timeout: float = 60.0
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    inference_concurrency: int = 1
    jwt_issuer: Optional[str] = None
    jwt_audience: Optional[str] = None


@dataclass(frozen=True)
class SourcePreflight:
    source_id: str
    kind: str
    path: Optional[Path]
    state: str
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "path": str(self.path) if self.path is not None else None,
            "state": self.state,
            "detail": self.detail,
        }


class RagConfig(dict):
    """Validated configuration with typed views and legacy mapping access."""

    def __init__(self, data: Mapping[str, Any], config_path: Path):
        super().__init__(data)
        self.config_path = config_path
        self.base_dir = config_path.parent
        self.pdf_source_models = _source_models(self, "pdf")
        self.json_source_models = _source_models(self, "json")
        self.extractor_model = _extractor_model(self.get("extractor"))
        self.generation_model = _generation_model(self.get("generation"))
        self.api_model = _api_model(self.get("api"))

    def to_json(self) -> str:
        return json.dumps(self, indent=2, sort_keys=True)


def _find_config(path: Optional[Path] = None) -> Path:
    if path is not None:
        candidate = Path(path)
    elif config_env := os.environ.get("RAG_CONFIG_PATH"):
        candidate = Path(config_env)
    else:
        cwd_config = Path.cwd() / DEFAULT_CONFIG_NAME
        candidate = cwd_config if cwd_config.exists() else Path(__file__).resolve().parents[2] / DEFAULT_CONFIG_NAME
    return candidate.expanduser().resolve()


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path, seen: Optional[set] = None) -> Dict[str, Any]:
    seen = set() if seen is None else seen
    if path in seen:
        chain = " -> ".join(str(p) for p in [*seen, path])
        raise ConfigError(f"Cyclic config overlay: {chain}")
    seen.add(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    extends = raw.pop("extends", None)
    if not extends:
        return raw
    parent = Path(str(extends)).expanduser()
    if not parent.is_absolute():
        parent = path.parent / parent
    return _deep_merge(_load_yaml(parent.resolve(), seen), raw)


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _apply_environment(cfg: Dict[str, Any]) -> None:
    if value := os.environ.get("RAG_VAULT_PATH"):
        cfg["vault_path"] = value
    if value := os.environ.get("RAG_INDEX_PATH"):
        cfg["index_path"] = value
    if value := os.environ.get("RAG_JSON_PATH"):
        sources = cfg.get("json_sources", [])
        if len(sources) > 1:
            raise ConfigError(
                "RAG_JSON_PATH is ambiguous when multiple json_sources are configured"
            )
        if sources:
            source = dict(sources[0])
            source["path"] = value
            cfg["json_sources"] = [source]
        else:
            cfg["json_sources"] = [{"path": value}]

    for env_name, source_type in (
        ("RAG_PDF_BOOKS_PATH", "book"),
        ("RAG_PDF_RESOURCES_PATH", "resource"),
    ):
        if value := os.environ.get(env_name):
            for source in cfg.get("pdf_sources", []):
                if source.get("type") == source_type:
                    source["path"] = value

    extractor = cfg.get("extractor")
    for env_name, field in _EXTRACTOR_ENV.items():
        if value := os.environ.get(env_name):
            if extractor is None:
                extractor = cfg.setdefault("extractor", {})
            extractor[field] = value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set, label: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"Unknown {label} key(s): {', '.join(unknown)}")


def _require_mapping(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return value


def _positive_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    try:
        value = int(cfg.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer") from exc
    if value <= 0:
        raise ConfigError(f"{key} must be greater than zero")
    return value


def _validate(cfg: Dict[str, Any]) -> None:
    _reject_unknown(cfg, _TOP_LEVEL_KEYS, "top-level config")
    if not isinstance(cfg.get("vault_path"), str) or not cfg["vault_path"].strip():
        raise ConfigError("vault_path must be a non-empty path")
    for key in ("collection_name", "embedding_model", "store"):
        if key in cfg and (not isinstance(cfg[key], str) or not cfg[key].strip()):
            raise ConfigError(f"{key} must be a non-empty string")
    for key in ("embedding_revision", "corpus_profile"):
        if key in cfg and not isinstance(cfg[key], str):
            raise ConfigError(f"{key} must be a string")
    if "embedding_dimension" in cfg:
        _positive_int(cfg, "embedding_dimension", 1)

    chunk_size = _positive_int(cfg, "chunk_max_chars", 1200)
    overlap = int(cfg.get("chunk_overlap_chars", 150))
    if overlap < 0 or overlap >= chunk_size:
        raise ConfigError("chunk_overlap_chars must be >= 0 and smaller than chunk_max_chars")
    for key, default in (
        ("embedding_batch_size", 16),
        ("markdown_workers", 1),
        ("pdf_workers", 1),
        ("rerank_fetch_k", 20),
        ("tag_fetch_k", 200),
        ("hybrid_fetch_k", 50),
        ("hybrid_rrf_k", 60),
    ):
        _positive_int(cfg, key, default)

    weights = cfg.get("hybrid_weights", [1.0, 1.0])
    if not isinstance(weights, (list, tuple)) or len(weights) != 2:
        raise ConfigError("hybrid_weights must contain [lexical_weight, dense_weight]")
    if any(float(weight) < 0 for weight in weights):
        raise ConfigError("hybrid_weights cannot contain negative values")

    try:
        prune_fraction = float(cfg.get("prune_max_fraction", 0.25))
        prune_chunks = int(cfg.get("prune_max_chunks", 10_000))
    except (TypeError, ValueError) as exc:
        raise ConfigError("prune thresholds must be numeric") from exc
    if not 0 <= prune_fraction <= 1:
        raise ConfigError("prune_max_fraction must be between 0 and 1")
    if prune_chunks < 0:
        raise ConfigError("prune_max_chunks must be >= 0")

    source_locations = {"markdown:vault": "vault_path"}
    for kind in ("pdf", "json"):
        sources = cfg.get(f"{kind}_sources", [])
        if not isinstance(sources, list):
            raise ConfigError(f"{kind}_sources must be a list")
        for index, source in enumerate(sources):
            location = f"{kind}_sources[{index}]"
            source = _require_mapping(source, location)
            _reject_unknown(source, _SOURCE_KEYS, location)
            if not isinstance(source.get("path"), str) or not source["path"].strip():
                raise ConfigError(f"{location}.path must be non-empty")
            for key in ("id", "type"):
                if key in source and (
                    not isinstance(source[key], str) or not source[key].strip()
                ):
                    raise ConfigError(f"{location}.{key} must be a non-empty string")
            source_id = source.get("id") or f"{kind}:{source.get('type') or index}"
            if source_id in source_locations:
                raise ConfigError(
                    f"Duplicate source id {source_id!r} at {source_locations[source_id]} "
                    f"and {location}"
                )
            source_locations[source_id] = location

    if "extractor" in cfg:
        extractor = _require_mapping(cfg["extractor"], "extractor")
        _reject_unknown(extractor, _EXTRACTOR_KEYS, "extractor")
    if "generation" in cfg:
        generation = _require_mapping(cfg["generation"], "generation")
        _reject_unknown(generation, _GENERATION_KEYS, "generation")
        if not generation.get("provider") or not generation.get("model"):
            raise ConfigError("generation.provider and generation.model are required")
        if int(generation.get("max_tokens", 1024)) <= 0:
            raise ConfigError("generation.max_tokens must be greater than zero")
    if "api" in cfg:
        api = _require_mapping(cfg["api"], "api")
        _reject_unknown(api, _API_KEYS, "api")
        port = int(api.get("port", 8000))
        if not 1 <= port <= 65535:
            raise ConfigError("api.port must be between 1 and 65535")
        if int(api.get("inference_concurrency", 1)) <= 0:
            raise ConfigError("api.inference_concurrency must be greater than zero")


def _resolve_paths(cfg: Dict[str, Any], base_dir: Path) -> None:
    for key in _PATH_KEYS:
        value = cfg.get(key)
        if value:
            cfg[key] = _resolve_path(value, base_dir)
    for list_name in ("pdf_sources", "json_sources"):
        for source in cfg.get(list_name, []):
            source["path"] = _resolve_path(source["path"], base_dir)
    for key, value in cfg.get("extractor", {}).items():
        if value:
            cfg["extractor"][key] = _resolve_path(value, base_dir)


def _source_models(cfg: Mapping[str, Any], kind: str) -> Tuple[SourceConfig, ...]:
    models = []
    for index, source in enumerate(cfg.get(f"{kind}_sources", [])):
        source_type = source.get("type")
        source_id = source.get("id") or f"{kind}:{source_type or index}"
        models.append(
            SourceConfig(
                source_id=source_id,
                path=Path(source["path"]),
                kind=kind,
                type=source_type,
                enabled=bool(source.get("enabled", True)),
            )
        )
    return tuple(models)


def _extractor_model(raw: Optional[Mapping[str, Any]]) -> Optional[ExtractorConfig]:
    if raw is None:
        return None
    values = {key: Path(value) if value else None for key, value in raw.items()}
    return ExtractorConfig(**values)


def _generation_model(raw: Optional[Mapping[str, Any]]) -> Optional[GenerationConfig]:
    return GenerationConfig(**raw) if raw else None


def _api_model(raw: Optional[Mapping[str, Any]]) -> ApiConfig:
    return ApiConfig(**(raw or {}))


def load_config(path: Optional[Path] = None) -> RagConfig:
    """Load, overlay, validate, and resolve one configuration profile."""
    config_path = _find_config(path)
    cfg = _load_yaml(config_path)
    _apply_environment(cfg)
    _validate(cfg)
    _resolve_paths(cfg, config_path.parent)
    return RagConfig(cfg, config_path)


def _path_state(path: Path) -> Tuple[str, str]:
    if not path.exists():
        return "missing", "path does not exist"
    if not os.access(path, os.R_OK):
        return "unreadable", "path is not readable"
    return "available", ""


def preflight_sources(config: RagConfig, include_extractor: bool = True) -> List[SourcePreflight]:
    """Return explicit availability states without mutating any source."""
    results = []
    vault = Path(config["vault_path"])
    state, detail = _path_state(vault)
    results.append(SourcePreflight("markdown:vault", "markdown", vault, state, detail))

    for kind, models in (
        ("pdf", config.pdf_source_models),
        ("json", config.json_source_models),
    ):
        if not models:
            results.append(SourcePreflight(f"{kind}:disabled", kind, None, "disabled", "not configured"))
        for source in models:
            if not source.enabled:
                results.append(SourcePreflight(source.source_id, kind, source.path, "disabled", "disabled by config"))
                continue
            state, detail = _path_state(source.path)
            results.append(SourcePreflight(source.source_id, kind, source.path, state, detail))

    if include_extractor:
        extractor = config.extractor_model
        if extractor is None:
            results.append(SourcePreflight("extractor:disabled", "extractor", None, "disabled", "not configured"))
        else:
            for field in ("books_path", "resources_path", "catalog_path"):
                path = getattr(extractor, field)
                if path is None:
                    results.append(SourcePreflight(f"extractor:{field}", "extractor", None, "disabled", "not configured"))
                else:
                    state, detail = _path_state(path)
                    results.append(SourcePreflight(f"extractor:{field}", "extractor", path, state, detail))
    return results
