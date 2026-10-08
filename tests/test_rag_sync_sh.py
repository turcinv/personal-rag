"""Behavioural tests for deploy/jetson/rag-sync.sh.

Drives the real bash script against a temp vault + temp local bare repo, with
the index and drift steps stubbed via env (RAG_SYNC_INDEX_CMD / _DRIFT_CMD) so
no docker, no model and no real index are touched. Proves: clean fast-forward
indexes once; no HEAD change skips indexing; a dirty tree aborts before pulling;
a diverged tree aborts without indexing."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "jetson" / "rag-sync.sh"


def _git(args, cwd):
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True, text=True)


def _make_vault_and_bare(root):
    bare = root / "bare.git"
    vault = root / "vault"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)],
                   check=True, capture_output=True, text=True)
    subprocess.run(["git", "init", "-b", "main", str(vault)],
                   check=True, capture_output=True, text=True)
    _git(["config", "user.email", "t@example.com"], vault)
    _git(["config", "user.name", "Test"], vault)
    (vault / "note.md").write_text("one\n", encoding="utf-8")
    _git(["add", "note.md"], vault)
    _git(["commit", "-m", "init"], vault)
    _git(["remote", "add", "origin", str(bare)], vault)
    _git(["push", "origin", "main"], vault)
    return bare, vault


def _advance_bare(bare, root):
    """Push a new commit into the bare repo via a throwaway clone."""
    other = root / "other"
    subprocess.run(["git", "clone", str(bare), str(other)],
                   check=True, capture_output=True, text=True)
    _git(["config", "user.email", "t@example.com"], other)
    _git(["config", "user.name", "Test"], other)
    (other / "note.md").write_text("one\ntwo\n", encoding="utf-8")
    _git(["commit", "-am", "advance"], other)
    _git(["push", "origin", "main"], other)
    shutil.rmtree(other)


def _rev(vault):
    return subprocess.run(["git", "-C", str(vault), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _run(vault, repo, index_marker, drift_argv=None, drift_rc=0):
    """Run the script with index/drift stubbed.

    The drift stub records its own argv to ``drift_argv`` (when given) and exits
    with ``drift_rc`` so tests can assert what the script passed to drift and how
    it reacts to a non-zero drift."""
    if drift_argv is not None:
        drift_cmd = f'printf "%s\\n" "$@" > {drift_argv}; exit {drift_rc}'
        drift_cmd = f'bash -c {_shq(drift_cmd)} _'
    else:
        drift_cmd = f"bash -c 'exit {drift_rc}'"
    env = {
        "RAG_SYNC_VAULT_DIR": str(vault),
        "RAG_SYNC_REPO_DIR": str(repo),
        "RAG_SYNC_REMOTE": "origin",
        "RAG_SYNC_BRANCH": "main",
        # Stub the heavy steps: index touches a marker, drift is scripted.
        "RAG_SYNC_INDEX_CMD": f"touch {index_marker}",
        "RAG_SYNC_DRIFT_CMD": drift_cmd,
        "PATH": os.environ["PATH"],
        "HOME": os.environ.get("HOME", str(repo)),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
    )


def _shq(s):
    return "'" + s.replace("'", "'\\''") + "'"


@pytest.fixture(autouse=True)
def _require_bash():
    if shutil.which("bash") is None:
        pytest.skip("bash not available")


def test_clean_ff_indexes_once(tmp_path):
    bare, vault = _make_vault_and_bare(tmp_path)
    _advance_bare(bare, tmp_path)  # vault is now behind origin -> ff moves HEAD
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "indexed"
    r = _run(vault, repo, marker)
    assert r.returncode == 0, r.stderr
    assert marker.exists(), "index step must run on a HEAD-moving fast-forward"


def test_drift_receives_skip_git(tmp_path):
    # The container drift step cannot do the git check; the script must pass
    # --skip-git and run the git comparison on the host instead.
    bare, vault = _make_vault_and_bare(tmp_path)
    _advance_bare(bare, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "indexed"
    argv = tmp_path / "drift_argv"
    r = _run(vault, repo, marker, drift_argv=argv)
    assert argv.exists(), r.stderr
    passed = argv.read_text().split("\n")
    assert "--skip-git" in passed, f"drift must receive --skip-git, got {passed}"


def test_no_change_skips_index(tmp_path):
    _bare, vault = _make_vault_and_bare(tmp_path)  # vault already at origin/main
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "indexed"
    r = _run(vault, repo, marker)
    assert r.returncode == 0, r.stderr
    assert not marker.exists(), "unchanged HEAD must skip the index step"
    assert "skipping index" in r.stdout


def test_dirty_tree_aborts_before_pull_and_index(tmp_path):
    bare, vault = _make_vault_and_bare(tmp_path)
    _advance_bare(bare, tmp_path)  # origin ahead, so a pull WOULD move HEAD
    (vault / "note.md").write_text("local uncommitted edit\n", encoding="utf-8")
    before = _rev(vault)
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "indexed"
    r = _run(vault, repo, marker)
    assert r.returncode != 0
    assert not marker.exists(), "dirty tree must not index"
    assert _rev(vault) == before, "dirty tree must not pull"


def test_diverged_tree_aborts_without_index(tmp_path):
    bare, vault = _make_vault_and_bare(tmp_path)
    _advance_bare(bare, tmp_path)  # origin has a commit the vault lacks
    # Give the vault its OWN divergent commit so a --ff-only pull cannot apply.
    (vault / "note.md").write_text("one\nlocal-divergent\n", encoding="utf-8")
    _git(["commit", "-am", "local divergent"], vault)
    repo = tmp_path / "repo"
    repo.mkdir()
    marker = tmp_path / "indexed"
    r = _run(vault, repo, marker)
    assert r.returncode != 0, "a diverged --ff-only pull must fail"
    assert not marker.exists(), "diverged tree must not index"
