# auth.py - Render-safe (SQLite + JWT)
from datetime import datetime, timedelta
from typing import Optional
import os
import re

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from db import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id
)

# ---------------- JWT Configuration ----------------
SECRET_KEY = os.environ.get("JWT_SECRET", "dev-secret-change-me")  # set on Render
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24h default

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ---------------- Password Utilities ----------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ---------------- Validation ----------------
def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# ---------------- User Authentication ----------------
def authenticate_user(username_or_email: str, password: str) -> Optional[dict]:
    """Authenticate user by username OR email."""
    user = get_user_by_username(username_or_email)
    if not user:
        user = get_user_by_email(username_or_email)

    if not user:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return user

def register_user(username: str, email: str, password: str) -> tuple[bool, str, Optional[int]]:
    """Optional helper if you want to register via auth module."""
    if not username or not email or not password:
        return False, "All fields are required", None

    if not validate_email(email):
        return False, "Invalid email format", None

    if len(password) < 6:
        return False, "Password must be at least 6 characters", None

    if len(username) < 3:
        return False, "Username must be at least 3 characters", None

    if get_user_by_username(username):
        return False, "Username already taken", None

    if get_user_by_email(email):
        return False, "Email already registered", None

    password_hash = get_password_hash(password)
    user_id = create_user(username, email, password_hash)

    if user_id:
        return True, "Registration successful", user_id
    return False, "Registration failed", None

# ---------------- Token Management ----------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()

    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})

    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Extract user from Bearer JWT token and return DB user dict."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(username)
    if not user:
        raise credentials_exception

    return user

def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    return current_user
