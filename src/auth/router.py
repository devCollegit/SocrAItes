# src/auth/router.py
"""Authentication API routes: register, login, OAuth callbacks, token refresh."""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr

from src.config.settings import get_settings
from src.db.database import get_connection, _now
from .jwt_handler import create_access_token, create_refresh_token, verify_token

logger = logging.getLogger("socraites.auth")
router = APIRouter(prefix="/auth", tags=["auth"])


# ─── Request / Response models ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ─── Password hashing (bcrypt) ────────────────────────────────────────────────

def _hash_password(password: str, salt: str = "") -> str:
    """Hash password using bcrypt for secure password storage."""
    import bcrypt
    # Combine email (salt) with password for additional uniqueness
    combined = f"{salt}{password}".encode()
    return bcrypt.hashpw(combined, bcrypt.gensalt()).decode()


def _verify_password(password: str, hashed: str, salt: str = "") -> bool:
    """Verify a password against a bcrypt hash."""
    import bcrypt
    combined = f"{salt}{password}".encode()
    return bcrypt.checkpw(combined, hashed.encode())


# ─── Database helpers ─────────────────────────────────────────────────────────

_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT,
    password_hash TEXT,
    provider    TEXT DEFAULT 'local',
    provider_id TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


def _ensure_users_table():
    """Create users table if it doesn't exist."""
    conn = get_connection()
    conn.executescript(_USERS_TABLE_SQL)
    conn.close()


def _get_user_by_email(email: str) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def _create_user(email: str, password_hash: str, name: Optional[str] = None, provider: str = "local", provider_id: Optional[str] = None) -> dict:
    user_id = str(uuid.uuid4())
    now = _now()
    conn = get_connection()
    conn.execute(
        "INSERT INTO users (id, email, name, password_hash, provider, provider_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (user_id, email, name, password_hash, provider, provider_id, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": user_id, "email": email, "name": name}


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """Register a new user with email and password."""
    _ensure_users_table()

    existing = _get_user_by_email(req.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )

    password_hash = _hash_password(req.password, salt=req.email)
    user = _create_user(email=req.email, password_hash=password_hash, name=req.name)

    access_token = create_access_token({"sub": user["id"], "email": user["email"]})
    refresh_token = create_refresh_token({"sub": user["id"], "email": user["email"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user["id"],
        email=user["email"],
    )


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """Authenticate with email and password."""
    _ensure_users_table()

    user = _get_user_by_email(req.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not _verify_password(req.password, user["password_hash"], salt=req.email):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    access_token = create_access_token({"sub": user["id"], "email": user["email"]})
    refresh_token = create_refresh_token({"sub": user["id"], "email": user["email"]})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user["id"],
        email=user["email"],
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(req: RefreshRequest):
    """Get a new access token using a refresh token."""
    payload = verify_token(req.refresh_token, token_type="refresh")
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_id = payload.get("sub")
    email = payload.get("email")

    access_token = create_access_token({"sub": user_id, "email": email})
    new_refresh_token = create_refresh_token({"sub": user_id, "email": email})

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        user_id=user_id,
        email=email,
    )


@router.get("/me")
async def get_me(user: dict = None):
    """Get current authenticated user info.
    
    Note: In production, use Depends(get_current_user).
    For now, this is a placeholder that works without auth enabled.
    """
    from .dependencies import get_current_user
    # This endpoint should be used with the dependency injection
    # get_current_user when auth is fully enabled
    return {"message": "Use this endpoint with ****** authentication"}
