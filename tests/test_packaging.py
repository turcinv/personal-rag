"""Dependency-ownership drift and egress/cloud-import confinement audits.

Offline, source-only checks (no install, no network):
  * every runtime dependency declared in pyproject is pinned in requirements.txt
  * ``httpx`` (the one outbound HTTP client) is used only in the generation layer
  * no cloud SDK (boto3 / google.cloud / gsutil) leaks into the codebase
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAG_SRC = REPO_ROOT / "src" / "rag"


def _normalize(name: str) -> str:
    # Strip version/marker/extras, lowercase, unify separators.
    base = re.split(r"[<>=!~;\[ ]", name.strip(), maxsplit=1)[0]
    return base.lower().replace("_", "-")


def _pyproject_runtime_deps() -> set:
    """Parse the [project].dependencies list without tomllib (Python 3.10-safe)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    start = text.index("dependencies = [")
    end = text.index("]", start)
    block = text[start:end]
    return {_normalize(m) for m in re.findall(r'"([^"]+)"', block)}


def _requirements_pins() -> set:
    names = set()
    for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*==", line)
        if match:
            names.add(_normalize(match.group(1)))
    return names


def test_pyproject_runtime_deps_are_pinned_in_requirements():
    """Every direct runtime dependency must appear pinned in the lockfile.

    torch is intentionally excluded from pyproject (installed per-platform), so
    it is not expected here. This catches a dependency added to pyproject but
    never compiled into requirements.txt (a reproducibility drift).
    """
    declared = _pyproject_runtime_deps()
    pinned = _requirements_pins()
    missing = sorted(dep for dep in declared if dep not in pinned)
    assert missing == [], f"pyproject deps not pinned in requirements.txt: {missing}"


def test_httpx_confined_to_generation_layer():
    """``httpx`` is the single intentional outbound path — only generation uses it."""
    offenders = []
    for path in sorted(RAG_SRC.rglob("*.py")):
        if path.parent.name == "generation":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bimport httpx\b|\bhttpx\.", text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"httpx used outside generation/: {offenders}"


def test_no_cloud_sdk_imports():
    """No cloud storage/compute SDK anywhere in the engine (audited clean 2026-07)."""
    banned = ("boto3", "google.cloud", "gsutil", "google.storage")
    offenders = []
    for path in sorted(RAG_SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            if token in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{token}")
    assert offenders == [], f"cloud SDK references found: {offenders}"
