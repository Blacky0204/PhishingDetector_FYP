# schemas.py - Pydantic v1 compatible version
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters')
        if not v.replace('_', '').isalnum():
            raise ValueError('Username can only contain letters, numbers, and underscores')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    username: Optional[str] = None
    user_id: Optional[int] = None


class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr
    created_at: str


class URLItem(BaseModel):
    url: str


class FrontendLogItem(BaseModel):
    url: str
    prediction: str
    explanation: List[str]
