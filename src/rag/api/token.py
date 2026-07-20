"""Mint an HS256 service token for the backend API (``rag-token``).

Signs a bearer token with the secret from ``RAG_API_JWT_SECRET`` so the internal
bots (Telegram RAG Bot, Wiki RAG Chatbot) — and tests/docs — can authenticate.
Defaults to a very long expiry, since these are internal service-to-service
credentials reachable only over Tailscale.

    RAG_API_JWT_SECRET=... rag-token --subject telegram-bot --expires-days 3650
"""

import argparse
import os
from datetime import datetime, timedelta, timezone

import jwt

from .auth import JWT_ALGORITHM, JWT_SECRET_ENV


def mint_token(subject: str = "service", expires_days: int = 3650) -> str:
    """Return a signed HS256 token for ``subject`` valid for ``expires_days``.

    The secret comes from ``RAG_API_JWT_SECRET`` only. Raises ``RuntimeError`` if
    it is unset (the value itself is never included in the message)."""
    secret = os.environ.get(JWT_SECRET_ENV)
    if not secret:
        raise RuntimeError(f"{JWT_SECRET_ENV} is not set in the environment")

    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "iat": now,
        "exp": now + timedelta(days=expires_days),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mint an HS256 service token for the personal-rag API "
        f"(reads the secret from ${JWT_SECRET_ENV}).",
    )
    parser.add_argument(
        "--subject", "--sub", dest="subject", default="service",
        help="Token subject (sub claim), e.g. a bot name (default: service).",
    )
    parser.add_argument(
        "--expires-days", type=int, default=3650, metavar="N",
        help="Days until the token expires (default: 3650).",
    )
    args = parser.parse_args()
    print(mint_token(args.subject, args.expires_days))


if __name__ == "__main__":
    main()
