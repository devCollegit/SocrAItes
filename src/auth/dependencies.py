# src/auth/dependencies.py
"""FastAPI dependency injection for authentication."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_handler import verify_token

# Optional bearer — does not raise 403 if token is absent
_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """Extract and validate the current user from the Authorization header.

    Raises 401 if the token is missing or invalid.
    Returns a dict with at least {"user_id": ..., "email": ...}.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials, token_type="access")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[dict]:
    """Same as get_current_user but returns None instead of raising 401.

    Useful for endpoints that work for both authenticated and anonymous users.
    """
    if credentials is None:
        return None

    payload = verify_token(credentials.credentials, token_type="access")
    if payload is None:
        return None

    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
    }
