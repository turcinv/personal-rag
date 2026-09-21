"""Generation-aware sync of published extractor artifacts to a remote host.

The legacy ``make sync-to-jetson`` rsynced the whole mutable output tree with a
bare ``python3`` config read, which could copy a half-written build, a ``current``
pointer ahead of its target, or a workstation-local lexical DB whose vector
generation ID does not match the remote index.

This module instead transfers exactly one *validated, immutable* artifact
generation and then flips the remote pointer last, so a remote consumer always
sees either the previous complete generation or the next one:

  1. copy ``generations/<id>/`` (immutable — safe to resend, never mutated)
  2. copy the compatibility alias symlinks (relative targets, transferred verbatim)
  3. copy the ``current`` symlink **last** — this is the activation

No ``--delete`` is used (old generations remain for rollback), and the lexical
index and Chroma store are never part of the transfer set.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from extractor.artifacts import ArtifactError, resolve_active
from rag.config import ConfigError, load_config

Command = Tuple[str, ...]
Runner = Callable[[Sequence[str]], None]

_ALIAS_NAMES = ("indexed", "resources.db", "text_output_books", "text_output_resources")


def plan_generation_sync(
    output_root: Path, host: str, remote_root: str
) -> List[Command]:
    """Return the ordered rsync commands that publish the active generation.

    ``resolve_active`` validates the manifest (schema, checksums, counts, SQLite
    integrity) before anything is planned, so an incomplete or tampered
    generation can never be selected for transfer.
    """
    if not host:
        raise ConfigError("A remote host is required (set JETSON_HOST).")
    if not remote_root:
        raise ConfigError("A remote output path is required (set JETSON_OUTPUT_PATH).")

    output_root = Path(output_root).expanduser().resolve()
    active = resolve_active(output_root)
    generation_id = active.name
    remote_root = remote_root.rstrip("/")

    commands: List[Command] = [
        (
            "rsync",
            "-a",
            "--",
            f"{active}/",
            f"{host}:{remote_root}/generations/{generation_id}/",
        )
    ]
    for name in _ALIAS_NAMES:
        alias = output_root / name
        if alias.is_symlink():
            commands.append(
                ("rsync", "-a", "--", str(alias), f"{host}:{remote_root}/")
            )
    commands.append(
        ("rsync", "-a", "--", str(output_root / "current"), f"{host}:{remote_root}/")
    )
    return commands


def _subprocess_runner(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def run_generation_sync(
    output_root: Path,
    host: str,
    remote_root: str,
    runner: Runner = _subprocess_runner,
) -> List[Command]:
    commands = plan_generation_sync(output_root, host, remote_root)
    for command in commands:
        runner(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-sync-generation",
        description="Transfer the active, validated artifact generation to a remote host.",
    )
    parser.add_argument("--config", help="Config path (overrides RAG_CONFIG_PATH)")
    parser.add_argument("--host", help="Remote host (default: JETSON_HOST env)")
    parser.add_argument(
        "--remote-path", help="Remote output root (default: JETSON_OUTPUT_PATH env)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the transfer plan and exit"
    )
    args = parser.parse_args()

    host = args.host or os.environ.get("JETSON_HOST", "")
    remote_root = args.remote_path or os.environ.get("JETSON_OUTPUT_PATH", "")

    try:
        config = load_config(args.config)
        extractor = config.extractor_model
        if extractor is None or extractor.output_path is None:
            raise ConfigError("The active profile has no extractor output path")
        commands = plan_generation_sync(extractor.output_path, host, remote_root)
    except ConfigError as exc:
        parser.error(str(exc))
    except ArtifactError as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "sync-to-jetson transfers a published, checksum-validated artifact "
            "generation, but none exists here. The per-stage flow "
            "(make extract/enrich/build-index ...) writes a flat indexed/ and never "
            "publishes a generation.\n"
            "  - To use sync-to-jetson: run a full `rag-pipeline` first (it publishes one).\n"
            "  - Or use the plain-rsync path (the default for the per-stage flow):\n"
            "      rsync -a <output>/indexed/ <host>:<remote>/indexed/\n"
            "      then `make jetson-index` on the host — idempotent, chunk IDs are "
            "content-hashed, so a partial transfer just resumes next run."
        )

    if args.dry_run:
        print("Generation sync plan (nothing transferred):")
        for command in commands:
            print("  " + " ".join(command))
        return

    for command in commands:
        print("==> " + " ".join(command))
        _subprocess_runner(command)
    print("Done. Run 'make build-jetson && make jetson-index' on the remote host.")


if __name__ == "__main__":
    main()
