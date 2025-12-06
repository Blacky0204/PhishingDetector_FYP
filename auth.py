# auth.py - UPDATED FOR SQLITE
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import re

from db import (
    create_user,
    get_user_by_username,
    get_user_by_email,
    get_user_by_id
)

# JWT Configuration
SECRET_KEY = "your-secret-key-for-jwt-tokens-keep-this-safe"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ============= PASSWORD UTILITIES =============
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

# ============= USER AUTHENTICATION =============
def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate user by username OR email"""
    # Try username first
    user = get_user_by_username(username)
    if not user:
        # Try email
        user = get_user_by_email(username)
    
    if not user:
        return None
    
    if not verify_password(password, user["password_hash"]):
        return None
    
    return user

def register_user(username: str, email: str, password: str) -> tuple[bool, str, Optional[int]]:
    """Register a new user"""
    # Validation
    if not username or not email or not password:
        return False, "All fields are required", None
    
    if not validate_email(email):
        return False, "Invalid email format", None
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters", None
    
    if len(username) < 3:
        return False, "Username must be at least 3 characters", None
    
    # Check if user exists
    if get_user_by_username(username):
        return False, "Username already taken", None
    
    if get_user_by_email(email):
        return False, "Email already registered", None
    
    # Create user
    password_hash = get_password_hash(password)
    user_id = create_user(username, email, password_hash)
    
    if user_id:
        return True, "Registration successful", user_id
    else:
        return False, "Registration failed", None

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# ============= JWT TOKEN MANAGEMENT =============
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_username(username)
    if user is None:
        raise credentials_exception
    
    return user

def get_current_active_user(current_user: dict = Depends(get_current_user)) -> dict:
    """Get current active user"""
    # Add any active user checks here if needed
    return current_user