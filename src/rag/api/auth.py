"""JWT authentication (HS256).

The shared secret is read from the ``RAG_API_JWT_SECRET`` environment variable
ONLY — never hardcoded, never with a default fallback. :func:`require_jwt` is a
FastAPI dependency that extracts and verifies a bearer token (signature + ``exp``)
and returns the decoded claims. Neither the token nor the secret is ever logged.
"""

import os
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_ALGORITHM = "HS256"
JWT_SECRET_ENV = "RAG_API_JWT_SECRET"
JWT_AUDIENCE_ENV = "RAG_API_JWT_AUDIENCE"
JWT_ISSUER_ENV = "RAG_API_JWT_ISSUER"
# HS256 keys shorter than this are brute-forceable; matches PyJWT's own warning.
MIN_SECRET_BYTES = 32

# Coarse-grained permissions. A token's ``scopes`` claim (list) restricts it to
# these actions; a token WITHOUT a scopes claim is unrestricted (full access),
# preserving compatibility with service tokens minted before scopes existed.
SCOPE_QUERY = "query"      # POST /query, GET /status
SCOPE_ANSWER = "answer"    # POST /answer (retrieval + LLM synthesis)
SCOPE_INDEX = "index"      # POST /index, GET /index/jobs/* (index administration)
ALL_SCOPES = (SCOPE_QUERY, SCOPE_ANSWER, SCOPE_INDEX)

# auto_error=False: we handle the missing/malformed header ourselves so that a
# missing Authorization header returns 401 (not FastAPI's default 403).
_bearer = HTTPBearer(auto_error=False)


def validate_secret_strength(secret: str | None) -> None:
    """Raise ``RuntimeError`` if the configured secret is too weak for HS256.

    Called at server startup (``rag-serve``) so a weak/short secret fails loudly
    before serving, rather than silently accepting brute-forceable tokens. An
    unset secret is left to the per-request 500 in :func:`_get_secret` so
    ``/health`` can still answer on a not-yet-configured deployment.
    """
    if secret is not None and len(secret.encode("utf-8")) < MIN_SECRET_BYTES:
        raise RuntimeError(
            f"{JWT_SECRET_ENV} is too short: HS256 needs at least "
            f"{MIN_SECRET_BYTES} bytes of secret material."
        )


def _get_secret() -> str:
    """Return the shared JWT secret from the environment.

    Raises 500 (misconfiguration, not 401) if the env var is unset — an unset
    secret is a server deployment error, not a client auth failure. The secret
    value is never included in the response or logs.
    """
    secret = os.environ.get(JWT_SECRET_ENV)
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="server auth not configured",
        )
    return secret


def require_jwt(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """FastAPI dependency guarding authenticated routes.

    Missing/malformed ``Authorization: Bearer <token>`` header → 401. Valid
    signature + unexpired → returns the decoded claims dict. Expired → 401
    "token expired"; any other verification failure → 401 "invalid token".
    """
    secret = _get_secret()

    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    # Audience/issuer are enforced only when configured (env set), so an
    # unconfigured deployment keeps validating signature + expiry exactly as
    # before. When set, PyJWT rejects a token whose aud/iss do not match.
    audience = os.environ.get(JWT_AUDIENCE_ENV) or None
    issuer = os.environ.get(JWT_ISSUER_ENV) or None
    decode_kwargs: dict[str, Any] = {"algorithms": [JWT_ALGORITHM]}
    if audience:
        decode_kwargs["audience"] = audience
    if issuer:
        decode_kwargs["issuer"] = issuer
    try:
        claims = jwt.decode(token, secret, **decode_kwargs)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return claims


def require_scope(scope: str):
    """Return a dependency that authenticates AND checks one permission scope.

    A token whose ``scopes`` claim omits ``scope`` gets 403. A token with **no**
    ``scopes`` claim is treated as unrestricted (full access) — this keeps
    long-lived service tokens minted before scopes existed working, while newly
    minted least-privilege tokens (e.g. a query-only bot) are enforced.
    """

    def dependency(claims: dict = Depends(require_jwt)) -> dict:
        token_scopes = claims.get("scopes")
        if token_scopes is None:
            return claims  # unscoped legacy token → full access
        if not isinstance(token_scopes, list) or scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"token is not authorized for the {scope!r} scope",
            )
        return claims

    return dependency
