"""
src/api/routers/auth.py
-----------------------
Authentication Router.

Endpoints:
  - POST /api/auth/register : Create new account (validates email uniqueness, hashes password)
  - POST /api/auth/login    : Authenticate credentials, issues signed JWT access token
  - GET  /api/auth/me       : Returns current authenticated user profile
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.api.dependencies import get_current_user
from src.core.database import get_pool
from src.core.logging_config import get_logger
from src.core.security import create_access_token, hash_password, verify_password

logger = get_logger("TripMateAI.api.auth")
router = APIRouter(prefix="/api/auth", tags=["Authentication"])

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$")


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Full name")
    email: str = Field(..., min_length=5, max_length=128, description="Valid unique email address")
    password: str = Field(..., min_length=6, max_length=128, description="Password (min 6 chars)")

    @field_validator("email")
    def validate_email_format(cls, v: str) -> str:
        clean = v.strip().lower()
        if not EMAIL_REGEX.match(clean):
            raise ValueError("Invalid email format")
        return clean


class LoginRequest(BaseModel):
    email: str = Field(..., description="Registered email address")
    password: str = Field(..., description="Account password")

    @field_validator("email")
    def clean_email(cls, v: str) -> str:
        return v.strip().lower()


class AuthResponse(BaseModel):
    success: bool
    message: str
    access_token: str | None = None
    token_type: str = "bearer"
    user: dict[str, Any] | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest):
    """
    Register a new user account.
    - Validates email uniqueness
    - Securely hashes password via PBKDF2-HMAC-SHA256
    - Generates persistent user record and signed JWT token
    """
    clean_email = request.email.strip().lower()
    pool = get_pool()

    try:
        with pool.connection() as conn:
            # Check if email is already registered
            existing = conn.execute(
                "SELECT id FROM users WHERE LOWER(email) = %s",
                (clean_email,),
            ).fetchone()

            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"An account with email '{clean_email}' already exists.",
                )

            user_id = f"user_{uuid.uuid4().hex[:12]}"
            hashed_pwd = hash_password(request.password)

            conn.execute(
                """
                INSERT INTO users (id, name, email, password_hash, created_at)
                VALUES (%s, %s, %s, %s, NOW())
                """,
                (user_id, request.name.strip(), clean_email, hashed_pwd),
            )

            # Initialize user profile
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, home_airport, updated_at)
                VALUES (%s, 'HYD', NOW())
                ON CONFLICT (user_id) DO NOTHING
                """,
                (user_id,),
            )

        token = create_access_token(
            data={"sub": user_id, "email": clean_email, "name": request.name.strip()}
        )
        logger.info("[AUTH] Registered new user id=%s email=%s", user_id, clean_email)

        return AuthResponse(
            success=True,
            message="User account created successfully.",
            access_token=token,
            user={"id": user_id, "name": request.name.strip(), "email": clean_email},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[AUTH] Registration error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user account.",
        )


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    """
    Authenticate a user with email and password.
    Returns a signed JWT bearer token on success.
    """
    clean_email = request.email.strip().lower()
    pool = get_pool()

    try:
        with pool.connection() as conn:
            user = conn.execute(
                "SELECT id, name, email, password_hash FROM users WHERE LOWER(email) = %s",
                (clean_email,),
            ).fetchone()

        if not user or not user["password_hash"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not verify_password(request.password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = create_access_token(
            data={"sub": user["id"], "email": user["email"], "name": user["name"]}
        )
        logger.info("[AUTH] User logged in: id=%s email=%s", user["id"], clean_email)

        return AuthResponse(
            success=True,
            message="Login successful.",
            access_token=token,
            user={"id": user["id"], "name": user["name"], "email": user["email"]},
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[AUTH] Login error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed.",
        )


@router.get("/me")
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)):
    """
    Retrieve the profile of the currently authenticated user from the verified JWT token.
    """
    pool = get_pool()
    with pool.connection() as conn:
        profile = conn.execute(
            """
            SELECT u.id, u.name, u.email, u.created_at,
                   p.home_airport, p.preferred_travel_style, p.dietary_preferences,
                   p.frequent_airlines, p.hotel_preferences, p.custom_notes
            FROM users u
            LEFT JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = %s
            """,
            (current_user["id"],),
        ).fetchone()

    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")

    return {
        "success": True,
        "user": dict(profile),
    }
