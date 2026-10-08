"""Unit tests for rag.drift — report-only source-vs-index drift detection.

Uses a fake embedding model and a real ChromaStore over a temp dir (the same
offline pattern as tests/test_indexing.py), so it runs with no GPU, no network,
and no real index. drift itself embeds nothing; the fake model is used only to
seed an in-sync index to compare against."""

import numpy as np
import pytest

from rag import drift
from rag.extractors import iter_sources
from rag.indexing import run_source
from rag.reconciliation import ReconciliationCatalog
from rag.store.chroma_store import ChromaStore
from pathlib import Path


class FakeModel:
    """Shape-correct, content-irrelevant stand-in for SentenceTransformer."""

    def encode(self, docs, normalize_embeddings=True, batch_size=16):
        return np.zeros((len(docs), 4))


def _write_note(vault: Path, name: str, body: str) -> None:
    (vault / "Knowledge").mkdir(parents=True, exist_ok=True)
    (vault / "Knowledge" / name).write_text(
        f"---\ntitle: {name}\ndomain: DevOps\n---\n# H\n{body}\n", encoding="utf-8"
    )


def _config(vault: Path, index_path: Path) -> dict:
    return {
        "vault_path": str(vault),
        "index_path": str(index_path),
        "collection_name": "drift",
        "exclude_dirs": [],
        "exclude_files": [],
        "markdown_workers": 1,
        "pdf_workers": 1,
        "chunk_max_chars": 1200,
        "chunk_overlap_chars": 150,
    }


def _seed_index(config: dict) -> ChromaStore:
    """Index the vault once with a fake model so the store matches the source."""
    vault = Path(config["vault_path"])
    index_path = Path(config["index_path"])
    store = ChromaStore(str(index_path), config["collection_name"])
    store.ensure(config["collection_name"])
    model = FakeModel()
    with ReconciliationCatalog.from_store(store, index_path) as catalog:
        for source in iter_sources(config, vault, 1200, 150):
            run_source(source, catalog, model, "cpu", 16, store)
    return store


def test_in_sync_reports_no_drift(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    sd = drift.check_source_drift(config)
    assert sd.state == "available"
    assert sd.missing_files == []
    assert sd.changed_files == []
    assert sd.stale_chunks == 0
    assert sd.drift is False


def test_added_file_reported_missing(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    # A note that exists in the source but was never indexed.
    _write_note(vault, "b.md", "fresh " * 300)
    sd = drift.check_source_drift(config)
    assert sd.missing_files == ["Knowledge/b.md"]
    assert sd.changed_files == []
    assert sd.drift is True


def test_changed_note_reported_and_leaves_stale(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    # Edit the body: chunks re-hash to new IDs, old ones become stale.
    _write_note(vault, "a.md", "totally different content " * 100)
    sd = drift.check_source_drift(config)
    assert sd.changed_files == ["Knowledge/a.md"]
    assert sd.stale_chunks > 0
    assert sd.drift is True


def test_non_git_vault_is_unknown_not_clean(monkeypatch):
    # Stub git so the test is independent of where the tmp dir lives: the
    # worktree is itself a git repo, so a real path-based check would otherwise
    # report in-tree. rev-parse --is-inside-work-tree returning non-"true" is
    # exactly what a non-git vault_path (e.g. a Jetson rsync target) produces.
    monkeypatch.setattr(drift, "_run_git", lambda args, cwd: None)
    git = drift.check_git_upstream(Path("/nonexistent/vault"))
    assert git.state == "unknown"
    assert git.is_git is False
    assert git.drift is False  # unknown never counts as drift
    assert "not a git work tree" in git.detail


def test_unknown_source_id_does_not_crash(tmp_path):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    sd = drift.check_source_drift(config, source_id="markdown:nope")
    assert sd.state == "unknown"
    assert sd.drift is False


def test_git_status_drift_logic():
    # behind > 0 => drift
    assert drift.GitStatus(state="ok", behind=3).drift is True
    # dirty > 0 => drift
    assert drift.GitStatus(state="ok", dirty=1).drift is True
    # ahead only, clean => no drift (host leads, but index can still match HEAD)
    assert drift.GitStatus(state="ok", ahead=5, behind=0, dirty=0).drift is False
    # in sync => no drift
    assert drift.GitStatus(state="ok", ahead=0, behind=0, dirty=0).drift is False
    # unknown => never drift
    assert drift.GitStatus(state="unknown", behind=99).drift is False


def test_main_exits_nonzero_on_drift(tmp_path, capsys, monkeypatch):
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)
    _write_note(vault, "b.md", "fresh " * 300)  # missing from index -> drift

    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 1
    assert "DRIFT DETECTED" in capsys.readouterr().out


def test_main_drift_wins_over_could_not_verify(tmp_path, capsys, monkeypatch):
    # Source drift present AND git unknown: drift (exit 1) must win over the
    # could-not-verify signal (exit 2) — an observed drift is the stronger fact.
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)
    _write_note(vault, "b.md", "fresh " * 300)  # missing from index -> drift

    monkeypatch.setattr(drift, "_run_git", lambda args, cwd: None)  # git unknown
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 1
    assert "DRIFT DETECTED" in capsys.readouterr().out


def test_main_exits_2_on_non_git_vault(tmp_path, capsys, monkeypatch):
    # In sync (no drift) but the git sub-check cannot run (non-git vault): the
    # command must NOT report a green "no drift" — exit 2, "COULD NOT VERIFY".
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    monkeypatch.setattr(drift, "_run_git", lambda args, cwd: None)  # not a git tree
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "COULD NOT VERIFY" in out
    assert "git check" in out


def test_main_exits_2_on_unavailable_source(tmp_path, capsys, monkeypatch):
    # The git sub-check is forced OK, but a requested source cannot be enumerated
    # (unknown source id stands in for the unavailable-vault case): exit 2.
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    monkeypatch.setattr(
        drift,
        "check_git_upstream",
        lambda vault_path, remote=drift.DEFAULT_REMOTE: drift.GitStatus(
            state="ok", is_git=True, ahead=0, behind=0, dirty=0
        ),
    )
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift", "--source", "markdown:nope"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "COULD NOT VERIFY" in out
    assert "markdown:nope" in out


def test_main_exits_zero_when_in_sync(tmp_path, capsys, monkeypatch):
    # In sync AND git verifiable (forced OK): the only path to a green exit 0.
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    monkeypatch.setattr(
        drift,
        "check_git_upstream",
        lambda vault_path, remote=drift.DEFAULT_REMOTE: drift.GitStatus(
            state="ok", is_git=True, ahead=0, behind=0, dirty=0
        ),
    )
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift", "--json"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 0


# ── §0b: --remote origin against a LOCAL bare repo (Jetson sync model) ────────

def _git(args, cwd):
    import subprocess
    subprocess.run(["git", "-C", str(cwd), *args], check=True,
                   capture_output=True, text=True)


def _init_vault_with_local_bare_remote(root):
    """Build a vault git repo + a local bare 'origin' it fast-forwards from.

    Mirrors the Jetson layout: the vault working copy has a single remote
    'origin' pointing at a local bare repo path (no network), exactly what
    `rag-drift --remote origin` must handle."""
    import subprocess

    bare = root / "career-knowledge-base.git"
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


def test_git_upstream_local_bare_remote_in_sync(tmp_path):
    _, vault = _init_vault_with_local_bare_remote(tmp_path)
    git = drift.check_git_upstream(vault, remote="origin")
    assert git.state == "ok"
    assert git.is_git is True
    assert git.remote == "origin"
    assert git.ahead == 0 and git.behind == 0 and git.dirty == 0
    assert git.drift is False


def test_git_upstream_local_bare_remote_behind(tmp_path):
    bare, vault = _init_vault_with_local_bare_remote(tmp_path)
    # Advance the bare repo via a second clone, so the vault is now behind origin.
    import subprocess
    other = tmp_path / "other"
    subprocess.run(["git", "clone", str(bare), str(other)],
                   check=True, capture_output=True, text=True)
    _git(["config", "user.email", "t@example.com"], other)
    _git(["config", "user.name", "Test"], other)
    (other / "note.md").write_text("one\ntwo\n", encoding="utf-8")
    _git(["commit", "-am", "advance"], other)
    _git(["push", "origin", "main"], other)

    git = drift.check_git_upstream(vault, remote="origin")
    assert git.state == "ok"
    assert git.behind == 1
    assert git.ahead == 0
    assert git.drift is True  # behind > 0 => drift


def test_git_upstream_local_bare_remote_dirty(tmp_path):
    _, vault = _init_vault_with_local_bare_remote(tmp_path)
    (vault / "note.md").write_text("dirty edit\n", encoding="utf-8")
    git = drift.check_git_upstream(vault, remote="origin")
    assert git.state == "ok"
    assert git.dirty == 1
    assert git.drift is True


def test_skip_git_reports_skipped_not_unknown(tmp_path, capsys, monkeypatch):
    # --skip-git: the git sub-check is reported "skipped" (state ok, not git),
    # so it neither contributes drift nor forces exit 2. In sync + skipped => 0.
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)

    # Prove it never touches git at all under --skip-git.
    def _boom(*a, **k):
        raise AssertionError("git must not be invoked under --skip-git")

    monkeypatch.setattr(drift, "_run_git", _boom)
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift", "--skip-git"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "skipped" in out
    assert "COULD NOT VERIFY" not in out


def test_skip_git_source_drift_still_exits_1(tmp_path, capsys, monkeypatch):
    # --skip-git does not mask a real source drift: the source sub-check still runs.
    vault = tmp_path / "vault"
    index_path = tmp_path / "idx"
    _write_note(vault, "a.md", "word " * 300)
    config = _config(vault, index_path)
    _seed_index(config)
    _write_note(vault, "b.md", "fresh " * 300)  # missing from index -> drift

    monkeypatch.setattr(drift, "_run_git", lambda args, cwd: None)
    monkeypatch.setattr(drift, "load_config", lambda: config)
    monkeypatch.setattr("sys.argv", ["rag-drift", "--skip-git"])
    with pytest.raises(SystemExit) as exc:
        drift.main()
    assert exc.value.code == 1
    assert "DRIFT DETECTED" in capsys.readouterr().out


# `from datetime import UTC` only exists on Python 3.11+. The Jetson image runs
# 3.10 (requires-python ">=3.10", ruff target-version py310), so such an import
# crashes the module on import there. drift.py used it once and broke rag-drift;
# this guard fails if it (or any future UTC import) comes back anywhere in src/.
def test_no_datetime_utc_import_in_src():
    import re

    repo_root = Path(__file__).resolve().parents[1]
    rag_src = repo_root / "src"
    # Matches `from datetime import UTC` with UTC as an imported name in any
    # position (e.g. "UTC, datetime" or "datetime, UTC"), but not a longer
    # identifier like MYUTC.
    pattern = re.compile(r"from\s+datetime\s+import\s+[^\n]*\bUTC\b")

    offenders = []
    for path in sorted(rag_src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(repo_root)))

    assert offenders == [], (
        "`from datetime import UTC` is Python 3.11+ only and breaks on the "
        f"Jetson's 3.10; use `timezone.utc` instead. Offenders: {offenders}"
    )
