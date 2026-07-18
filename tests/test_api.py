"""Offline unit tests for the FastAPI backend (rag.api).

Runs in the offline unit suite: no network, no real model download, no GPU. The
real app loads MiniLM + Chroma + a cross-encoder in its lifespan; these tests
NEVER enter that lifespan. Instead a fake RAG state dict is injected via
``app.dependency_overrides[get_rag_state]`` (TestClient is used WITHOUT the
context-manager form, so startup/shutdown never run), mirroring the fake-model /
fake-collection approach in ``test_indexing.py`` and ``test_query_search.py``.

This file grows across the backend-API phases. This slice covers:
  * auth (JWT / bearer contract) on a protected route
  * GET /health (unauthenticated liveness)
  * GET /status (authenticated collection report)
Later slices append POST /query and POST /index tests.
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


class FakeCollection:
    """Stand-in for a Chroma collection — only /status's needs (.name, .count())."""

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
        "collection": FakeCollection(name="obsidian_markdown", count=1234),
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
    """/status reads .count() live off the (fake) collection, not a cached value."""
    fake_state["collection"] = FakeCollection(name="custom_coll", count=0)
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
