"""Offline unit tests for the FastAPI backend (rag.api).

Runs in the offline unit suite: no network, no real model download, no GPU. The
real app loads MiniLM + Chroma + a cross-encoder in its lifespan; these tests
NEVER enter that lifespan. Instead a fake RAG state dict is injected via
``app.dependency_overrides[get_rag_state]`` (TestClient is used WITHOUT the
context-manager form, so startup/shutdown never run), mirroring the fake-model /
fake-collection approach in ``test_indexing.py`` and ``test_query_search.py``.

This file grows across the backend-API phases. Covered so far:
  * auth (JWT / bearer contract) on a protected route
  * GET /health (unauthenticated liveness)
  * GET /status (authenticated collection report)
  * POST /query (envelope, n_results cap, filter->build_where, rerank flag)
  * POST /index + GET /index/jobs/{id} (202, concurrency 409, success/failure, 404, auth)
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from rag.api.app import app
from rag.api.auth import JWT_ALGORITHM, JWT_SECRET_ENV
from rag.api.deps import get_rag_state

# ── shared test constants / fakes ──────────────────────────────────────────────

# ≥32 bytes so PyJWT does not emit InsecureKeyLengthWarning for HS256.
SECRET = "unit-test-jwt-secret-key-at-least-32-bytes-long"
WRONG_SECRET = "some-other-secret-key-also-well-over-32-bytes-long"


class FakeStore:
    """Stand-in for a RetrievalStore — only /status's needs (.name, .count())."""

    def __init__(self, name="obsidian_markdown", count=1234):
        self.name = name
        self._count = count

    def count(self):
        return self._count


def _mint(secret=SECRET, *, sub="unit-test", expires_delta=timedelta(hours=1)):
    """Sign an HS256 token directly (full control over secret + expiry)."""
    now = datetime.now(timezone.utc)
    payload = {"sub": sub, "iat": now, "exp": now + expires_delta}
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


# ── fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_state():
    """The fake app.state.rag dict routes read via get_rag_state."""
    return {
        "config": {},
        "model": object(),
        "store": FakeStore(name="obsidian_markdown", count=1234),
        "reranker": object(),
        "embedding_model": "fake-embed",
        "reranker_model": "fake-rerank",
    }


@pytest.fixture
def client(fake_state):
    """TestClient with get_rag_state overridden — the heavy lifespan never runs.

    Not used as a context manager, so Starlette does not trigger startup/shutdown
    (which would load the real model). Overrides are cleared on teardown so cases
    never leak into one another.
    """
    app.dependency_overrides[get_rag_state] = lambda: fake_state
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def jwt_secret(monkeypatch):
    """Set the shared JWT secret in the env (read at request time by auth._get_secret)."""
    monkeypatch.setenv(JWT_SECRET_ENV, SECRET)
    return SECRET


# ── GET /health (unauthenticated) ───────────────────────────────────────────────


def test_health_no_auth_returns_ok(client):
    """/health is a liveness probe: no Authorization header, no model touch."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_ignores_secret_and_token(client, jwt_secret):
    """/health stays 200 even with the secret set and no/garbage token."""
    assert client.get("/health").status_code == 200
    assert client.get("/health", headers=_auth("garbage")).status_code == 200


# ── auth contract on a protected route (/status) ────────────────────────────────


def test_status_missing_auth_header_401(client, jwt_secret):
    resp = client.get("/status")
    assert resp.status_code == 401


def test_status_malformed_bearer_token_401(client, jwt_secret):
    resp = client.get("/status", headers=_auth("not-a-real-jwt"))
    assert resp.status_code == 401


def test_status_wrong_secret_token_401(client, jwt_secret):
    """A well-formed token signed with the wrong secret fails signature check."""
    token = _mint(secret=WRONG_SECRET)
    resp = client.get("/status", headers=_auth(token))
    assert resp.status_code == 401


def test_status_expired_token_401(client, jwt_secret):
    token = _mint(expires_delta=timedelta(hours=-1))  # exp in the past
    resp = client.get("/status", headers=_auth(token))
    assert resp.status_code == 401
    assert resp.json()["detail"] == "token expired"


def test_status_empty_bearer_credential_401(client, jwt_secret):
    """`Authorization: Bearer` with no token → treated as missing credentials."""
    resp = client.get("/status", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


# ── GET /status (valid auth) ────────────────────────────────────────────────────


def test_status_valid_token_reports_fake_state(client, jwt_secret):
    token = _mint()
    resp = client.get("/status", headers=_auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "collection": "obsidian_markdown",
        "count": 1234,
        "embedding_model": "fake-embed",
        "reranker_model": "fake-rerank",
    }


def test_status_reflects_live_collection_count(client, fake_state):
    """/status reads .count() live off the (fake) store, not a cached value."""
    fake_state["store"] = FakeStore(name="custom_coll", count=0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(JWT_SECRET_ENV, SECRET)
        resp = client.get("/status", headers=_auth(_mint()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["collection"] == "custom_coll"
    assert body["count"] == 0


# ── server misconfiguration: secret unset ───────────────────────────────────────


def test_status_secret_unset_is_500(client, monkeypatch):
    """Unset RAG_API_JWT_SECRET is a server config error (500), not a 401.

    auth._get_secret reads the env at request time, so deleting the var here is
    enough. A valid-looking bearer token is sent to prove the 500 comes from the
    missing secret, not from the missing header.
    """
    monkeypatch.delenv(JWT_SECRET_ENV, raising=False)
    token = _mint()  # minted against SECRET, but the server has no secret to check
    resp = client.get("/status", headers=_auth(token))
    assert resp.status_code == 500


# ── POST /query ─────────────────────────────────────────────────────────────────
#
# The route does `from ... import query as rag_query` then calls
# `rag_query.search(...)` / `rag_query.build_where(...)`. We patch `search` on the
# shared `rag.query` module (the object `rag_query` refers to) so the route calls
# our recorder — no real model/collection is touched — while the REAL `build_where`
# still runs, letting us assert the exact where-dict the route produced.

import rag.query as rag_query_mod  # noqa: E402  (kept with the /query section)


class SearchRecorder:
    """Captures the kwargs the route passes to search() and returns canned records."""

    def __init__(self, records):
        self.records = records
        self.calls = []

    def __call__(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return self.records

    @property
    def last(self):
        return self.calls[-1]


def _records(with_rerank_score=False):
    rec = {
        "document": "some chunk text",
        "metadata": {"title": "T", "path": "p.md", "domain": "DevOps"},
        "distance": 0.12,
        "rank": 1,
    }
    if with_rerank_score:
        rec["rerank_score"] = 9.5
    return [rec]


@pytest.fixture
def patch_search(monkeypatch):
    """Install a SearchRecorder over rag.query.search; return a factory.

    Call the returned factory with the records you want search() to yield; it
    swaps in a fresh recorder and hands it back so the test can inspect .last.
    monkeypatch tears the patch down automatically after the test.
    """

    def install(records):
        rec = SearchRecorder(records)
        monkeypatch.setattr(rag_query_mod, "search", rec)
        return rec

    return install


def test_query_valid_returns_envelope(client, jwt_secret, patch_search, fake_state):
    rec = patch_search(_records(with_rerank_score=True))
    resp = client.post(
        "/query", headers=_auth(_mint()), json={"query": "how do I do X"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "how do I do X"
    assert body["count"] == 1
    assert body["reranked"] is True
    assert isinstance(body["results"], list) and len(body["results"]) == 1
    r = body["results"][0]
    assert r["document"] == "some chunk text"
    assert r["metadata"]["path"] == "p.md"
    assert r["distance"] == 0.12
    assert r["rank"] == 1
    # app.state's model/store/config (the injected fakes) were forwarded to search()
    assert rec.last["model"] is fake_state["model"]
    assert rec.last["store"] is fake_state["store"]
    assert rec.last["config"] is fake_state["config"]


def test_query_defaults_forwarded(client, jwt_secret, patch_search):
    """Body with only `query` → search() gets n_results=8, rerank=True, filters=None."""
    rec = patch_search(_records())
    resp = client.post("/query", headers=_auth(_mint()), json={"query": "q"})
    assert resp.status_code == 200
    assert rec.last["n_results"] == 8
    assert rec.last["rerank"] is True
    assert rec.last["filters"] is None


@pytest.mark.parametrize(
    "n_results,expected",
    [(0, 422), (51, 422), (1, 200), (50, 200)],
)
def test_query_n_results_cap(client, jwt_secret, patch_search, n_results, expected):
    patch_search(_records())
    resp = client.post(
        "/query",
        headers=_auth(_mint()),
        json={"query": "q", "n_results": n_results},
    )
    assert resp.status_code == expected


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t "])
def test_query_empty_or_whitespace_query_422(client, jwt_secret, patch_search, bad_query):
    patch_search(_records())
    resp = client.post("/query", headers=_auth(_mint()), json={"query": bad_query})
    assert resp.status_code == 422


def test_query_filters_map_to_build_where(client, jwt_secret, patch_search):
    rec = patch_search(_records())
    resp = client.post(
        "/query",
        headers=_auth(_mint()),
        json={"query": "q", "filters": {"domain": "DevOps", "type": "book"}},
    )
    assert resp.status_code == 200
    assert rec.last["filters"] == {
        "$and": [
            {"domain": {"$eq": "DevOps"}},
            {"type": {"$eq": "book"}},
        ]
    }


def test_query_no_filters_passes_none(client, jwt_secret, patch_search):
    rec = patch_search(_records())
    resp = client.post("/query", headers=_auth(_mint()), json={"query": "q"})
    assert resp.status_code == 200
    assert rec.last["filters"] is None
    assert rec.last["tags"] is None      # no filters block -> tags None too


def test_query_status_filter_maps_to_build_where(client, jwt_secret, patch_search):
    """`status` is a native $eq clause produced by the REAL build_where."""
    rec = patch_search(_records())
    resp = client.post(
        "/query",
        headers=_auth(_mint()),
        json={"query": "q", "filters": {"status": "processed"}},
    )
    assert resp.status_code == 200
    assert rec.last["filters"] == {"status": {"$eq": "processed"}}


def test_query_tags_forwarded_to_search_not_where(client, jwt_secret, patch_search):
    """`tags` is a post-filter: it reaches search() as a list and never becomes a
    where clause (filters stays None when only tags are given)."""
    rec = patch_search(_records())
    resp = client.post(
        "/query",
        headers=_auth(_mint()),
        json={"query": "q", "filters": {"tags": ["devops", "ci"]}},
    )
    assert resp.status_code == 200
    assert rec.last["tags"] == ["devops", "ci"]
    assert rec.last["filters"] is None


def test_query_status_and_tags_together(client, jwt_secret, patch_search):
    """status rides in the where-dict (with domain), tags go to search()."""
    rec = patch_search(_records())
    resp = client.post(
        "/query",
        headers=_auth(_mint()),
        json={
            "query": "q",
            "filters": {"domain": "DevOps", "status": "processed", "tags": ["devops"]},
        },
    )
    assert resp.status_code == 200
    assert rec.last["filters"] == {
        "$and": [
            {"domain": {"$eq": "DevOps"}},
            {"status": {"$eq": "processed"}},
        ]
    }
    assert rec.last["tags"] == ["devops"]


def test_query_rerank_true_with_scores_reports_reranked(client, jwt_secret, patch_search):
    rec = patch_search(_records(with_rerank_score=True))
    resp = client.post(
        "/query", headers=_auth(_mint()), json={"query": "q", "rerank": True}
    )
    assert resp.status_code == 200
    assert rec.last["rerank"] is True
    assert resp.json()["reranked"] is True


def test_query_rerank_false_reports_not_reranked(client, jwt_secret, patch_search):
    rec = patch_search(_records(with_rerank_score=False))
    resp = client.post(
        "/query", headers=_auth(_mint()), json={"query": "q", "rerank": False}
    )
    assert resp.status_code == 200
    assert rec.last["rerank"] is False
    assert resp.json()["reranked"] is False


def test_query_rerank_true_but_no_scores_reports_not_reranked(
    client, jwt_secret, patch_search
):
    """reranked is False when requested but no record carries a rerank_score
    (e.g. empty dense pool), matching the handler's `any(...)` guard."""
    patch_search(_records(with_rerank_score=False))
    resp = client.post(
        "/query", headers=_auth(_mint()), json={"query": "q", "rerank": True}
    )
    assert resp.status_code == 200
    assert resp.json()["reranked"] is False


def test_query_no_token_401_before_search(client, jwt_secret, patch_search):
    rec = patch_search(_records())
    resp = client.post("/query", json={"query": "q"})
    assert resp.status_code == 401
    assert rec.calls == []  # auth rejected before any retrieval


def test_query_invalid_token_401_before_search(client, jwt_secret, patch_search):
    rec = patch_search(_records())
    resp = client.post("/query", headers=_auth("garbage"), json={"query": "q"})
    assert resp.status_code == 401
    assert rec.calls == []


# ── POST /index + GET /index/jobs/{id} ──────────────────────────────────────────
#
# The job manager (rag.api.jobs.manager) is a PROCESS-GLOBAL singleton, so a
# leftover "running" job would leak into later tests and cause spurious 409s. The
# `indexer` fixture (below) both (a) monkeypatches the subprocess seam
# `rag.api.jobs._spawn_indexer` with a fake process — NO real subprocess, NO real
# indexing — and (b) resets the manager registry before and after each test and
# releases any still-blocked fake process so no monitor thread lingers.

import threading  # noqa: E402
from pathlib import Path  # noqa: E402

import rag.api.jobs as jobs  # noqa: E402


class FakeProc:
    """Stand-in for subprocess.Popen: .wait() blocks on an Event, .returncode set.

    With ``release_immediately`` the wait returns at once (normal fast tests);
    otherwise a test holds the process in the "running" window until it calls
    ``release()`` — deterministic, no sleeps.
    """

    def __init__(self, returncode: int, release_immediately: bool, exc: Exception | None = None):
        self._returncode = returncode
        self._exc = exc
        self._event = threading.Event()
        if release_immediately:
            self._event.set()
        self.returncode = None
        self.pid = 4242

    def wait(self) -> int:
        self._event.wait()
        if self._exc is not None:
            raise self._exc
        self.returncode = self._returncode
        return self._returncode

    def release(self) -> None:
        self._event.set()


class _IndexerControl:
    """Test handle over the fake spawn: configure exit code / blocking / log line
    and inspect the fake processes that were spawned."""

    def __init__(self):
        self._cfg = {"returncode": 0, "block": False, "log_line": None, "exc": None}
        self.procs: list[FakeProc] = []

    def configure(self, **kw):
        self._cfg.update(kw)

    def _spawn(self, log_path):
        proc = FakeProc(
            self._cfg["returncode"],
            release_immediately=not self._cfg["block"],
            exc=self._cfg["exc"],
        )
        if self._cfg["log_line"] is not None:
            Path(log_path).write_text(self._cfg["log_line"], encoding="utf-8")
        proc.log_path = str(log_path)
        self.procs.append(proc)
        return proc


def _reset_manager():
    with jobs.manager._lock:
        jobs.manager._jobs.clear()
        jobs.manager._threads.clear()


@pytest.fixture
def indexer(monkeypatch, tmp_path):
    """Fake subprocess seam + manager reset. Yields an _IndexerControl.

    Job logs are directed into tmp_path via RAG_LOG_PATH so the route's log-dir
    mkdir never writes into the repo.
    """
    monkeypatch.setenv("RAG_LOG_PATH", str(tmp_path / "logs" / "rag.log"))
    ctl = _IndexerControl()
    monkeypatch.setattr(jobs, "_spawn_indexer", ctl._spawn)
    _reset_manager()
    try:
        yield ctl
    finally:
        # Release any process still blocked in wait(), drain its monitor thread,
        # then clear the registry so no job leaks into the next test.
        for p in ctl.procs:
            p.release()
        for jid in list(jobs.manager._threads):
            jobs.manager.join(jid, timeout=2.0)
        _reset_manager()


def test_index_start_returns_202_no_path_leak(client, jwt_secret, indexer):
    resp = client.post("/index", headers=_auth(_mint()))
    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"]
    assert body["status"] == "running"  # impl inserts directly as running (no queued phase)
    # The server-side log/host path must never appear in the response body.
    assert "log_path" not in body and "path" not in body
    assert not any(isinstance(v, str) and "index-" in v and ".log" in v for v in body.values())


def test_index_job_succeeds_after_join(client, jwt_secret, indexer):
    indexer.configure(returncode=0, block=False)
    jid = client.post("/index", headers=_auth(_mint())).json()["job_id"]
    jobs.manager.join(jid, timeout=2.0)

    resp = client.get(f"/index/jobs/{jid}", headers=_auth(_mint()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["started"] and body["finished"]
    assert body["returncode"] == 0
    assert body["error"] is None


def test_index_second_run_409_then_freed(client, jwt_secret, indexer):
    # First run stays in the "running" window (its .wait() blocks until released).
    indexer.configure(block=True)
    r1 = client.post("/index", headers=_auth(_mint()))
    assert r1.status_code == 202
    jid = r1.json()["job_id"]

    # Second POST while the first is running → 409, and no second process spawned.
    r2 = client.post("/index", headers=_auth(_mint()))
    assert r2.status_code == 409
    assert len(indexer.procs) == 1

    # Let the first finish; the active guard should clear.
    indexer.procs[0].release()
    jobs.manager.join(jid, timeout=2.0)

    indexer.configure(block=False)
    r3 = client.post("/index", headers=_auth(_mint()))
    assert r3.status_code == 202
    assert r3.json()["job_id"] != jid


def test_index_job_failed_reports_error(client, jwt_secret, indexer):
    """A nonzero exit (simulating the indexer's 0-files anti-wipe RuntimeError)
    surfaces as status=failed with a non-empty error from the log tail."""
    indexer.configure(
        returncode=1,
        block=False,
        log_line="RuntimeError: refusing to prune: 0 files found in every source",
    )
    jid = client.post("/index", headers=_auth(_mint())).json()["job_id"]
    jobs.manager.join(jid, timeout=2.0)

    resp = client.get(f"/index/jobs/{jid}", headers=_auth(_mint()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"
    assert body["returncode"] == 1
    assert body["error"] and "0 files" in body["error"]
    assert "log_path" not in body


def test_index_job_unknown_404(client, jwt_secret, indexer):
    resp = client.get("/index/jobs/does-not-exist", headers=_auth(_mint()))
    assert resp.status_code == 404


def test_index_start_requires_auth_no_spawn(client, jwt_secret, indexer):
    assert client.post("/index").status_code == 401
    assert client.post("/index", headers=_auth("garbage")).status_code == 401
    # Auth rejected before manager.start → no process was ever spawned.
    assert indexer.procs == []


def test_index_job_status_requires_auth(client, jwt_secret, indexer):
    assert client.get("/index/jobs/whatever").status_code == 401
    assert client.get("/index/jobs/whatever", headers=_auth("garbage")).status_code == 401


# ── jobs.py robustness regressions (commit 32a9a3a) ─────────────────────────────


def test_index_monitor_crash_fails_job_and_unblocks(client, jwt_secret, indexer):
    """Regression: a crashing monitor (proc.wait() raises) must NOT leave the job
    stuck 'running' — that would 409 every future reindex forever. It must
    force-transition to 'failed', and a subsequent POST must be allowed again."""
    indexer.configure(exc=RuntimeError("boom in wait"), block=False)
    jid = client.post("/index", headers=_auth(_mint())).json()["job_id"]
    jobs.manager.join(jid, timeout=2.0)

    resp = client.get(f"/index/jobs/{jid}", headers=_auth(_mint()))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "failed"          # NOT stuck at "running"
    assert body["finished"]                     # finish timestamp recorded
    assert body["error"] and "monitor crashed" in body["error"]

    # Recovery: the concurrency guard is no longer locked at 409.
    indexer.configure(exc=None, block=False)
    r2 = client.post("/index", headers=_auth(_mint()))
    assert r2.status_code == 202
    assert r2.json()["job_id"] != jid


def test_index_failed_error_is_single_trimmed_line(client, jwt_secret, indexer):
    """Regression: the HTTP error is the LAST non-empty log line (≤300 chars),
    not the raw multi-line tail — so tracebacks and container paths don't leak."""
    multiline_log = (
        "Traceback (most recent call last):\n"
        '  File "/app/src/rag/indexer.py", line 98, in main\n'
        "    raise RuntimeError(...)\n"
        "RuntimeError: Every source reported 0 files while the index holds chunks. "
        "Refusing to prune the entire collection\n"
    )
    indexer.configure(returncode=1, block=False, log_line=multiline_log)
    jid = client.post("/index", headers=_auth(_mint())).json()["job_id"]
    jobs.manager.join(jid, timeout=2.0)

    body = client.get(f"/index/jobs/{jid}", headers=_auth(_mint())).json()
    assert body["status"] == "failed"
    err = body["error"]
    assert "Every source reported 0 files" in err  # (i) the RuntimeError reason
    assert "Refusing to prune" in err
    assert "\n" not in err                          # (ii) single line
    assert "/app/" not in err                        # (iii) no container path
    assert "Traceback" not in err                    #      no traceback header
    assert len(err) <= 300                            # (iv) capped
