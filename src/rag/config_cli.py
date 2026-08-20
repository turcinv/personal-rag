"""Configuration inspection and source-preflight CLI."""

import argparse
import json

from .config import ConfigError, load_config, preflight_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-config",
        description="Validate configuration and report resolved source paths.",
    )
    parser.add_argument("--config", help="Config path (overrides RAG_CONFIG_PATH)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any enabled source is missing or unreadable",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        parser.error(str(exc))

    checks = preflight_sources(config)
    if args.as_json:
        print(
            json.dumps(
                {
                    "config_path": str(config.config_path),
                    "collection_name": config.get("collection_name"),
                    "store": config.get("store"),
                    "sources": [check.as_dict() for check in checks],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Config: {config.config_path}")
        print(f"Collection: {config.get('collection_name')}")
        print(f"Store: {config.get('store')}")
        print("Sources:")
        for check in checks:
            path = str(check.path) if check.path is not None else "-"
            suffix = f" ({check.detail})" if check.detail else ""
            print(f"  {check.state:10s} {check.source_id:28s} {path}{suffix}")

    unsafe = [check for check in checks if check.state in {"missing", "unreadable"}]
    if args.strict and unsafe:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
