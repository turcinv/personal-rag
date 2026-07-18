"""JWT authentication (HS256).

The shared secret is read from the ``RAG_API_JWT_SECRET`` environment variable
ONLY — never hardcoded, never with a default fallback. :func:`require_jwt` is a
FastAPI dependency that extracts and verifies a bearer token (signature + ``exp``)
and returns the decoded claims. Neither the token nor the secret is ever logged.
"""

import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_ALGORITHM = "HS256"
JWT_SECRET_ENV = "RAG_API_JWT_SECRET"

# auto_error=False: we handle the missing/malformed header ourselves so that a
# missing Authorization header returns 401 (not FastAPI's default 403).
_bearer = HTTPBearer(auto_error=False)


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
    try:
        claims = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
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
