"""
src/api/dependencies.py
-----------------------
FastAPI Authentication Dependencies.

Provides:
  - get_current_user: Strictly requires a valid JWT Bearer token.
                      Used for user profile, memory management, and private actions.
  - get_optional_current_user: Returns authenticated user if Bearer token is provided,
                               otherwise falls back to "anonymous" or caller-specified ID.
                               Ensures backwards compatibility for unauthenticated frontend demo.
"""
from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.core.security import decode_access_token
from src.core.logging_config import get_logger

logger = get_logger("TripMateAI.api.auth")

# HTTP Bearer scheme (handles Authorization: Bearer <token>)
security_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security_bearer)],
) -> dict[str, Any]:
    """
    Enforces JWT authentication on a route.
    Returns the user claims dict: {"id": user_id, "email": email, "name": name}.
    Raises HTTP 401 if token is missing, invalid, or expired.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed token: missing subject claim.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "id": user_id,
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
    }


async def get_optional_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security_bearer)],
) -> dict[str, Any] | None:
    """
    Optional authentication: returns user dict if Bearer token is present and valid,
    otherwise returns None without raising an exception.
    """
    if not credentials or not credentials.credentials:
        return None

    payload = decode_access_token(credentials.credentials)
    if not payload or not payload.get("sub"):
        return None

    return {
        "id": payload.get("sub"),
        "email": payload.get("email", ""),
        "name": payload.get("name", ""),
    }
