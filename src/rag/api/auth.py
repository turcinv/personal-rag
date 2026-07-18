"""JWT authentication (HS256).

Phase 2 placeholder: :func:`require_jwt` is defined and importable so routes can
depend on it now, but signature/expiry verification is a TODO filled in Phase 5.
The secret is read from the ``RAG_API_JWT_SECRET`` environment variable only —
never hardcoded.
"""

import os

JWT_ALGORITHM = "HS256"
JWT_SECRET_ENV = "RAG_API_JWT_SECRET"


def _get_secret() -> str | None:
    """Return the shared JWT secret from the environment (or None if unset)."""
    return os.environ.get(JWT_SECRET_ENV)


def require_jwt() -> dict:
    """FastAPI dependency guarding authenticated routes.

    TODO(Phase 5): read the ``Authorization: Bearer <token>`` header, verify the
    HS256 signature and ``exp`` against ``RAG_API_JWT_SECRET`` with pyjwt, and
    return the decoded claims (raising 401 on any failure). For now this is a
    no-op placeholder so routes can declare the dependency.
    """
    return {}
