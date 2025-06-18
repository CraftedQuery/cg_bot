"""
auth.py - Authentication and user management for the RAG chatbot
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .models import User, UserCreate, TokenData
from .config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    BASE_DIR,
    AAD_TENANT_ID,
    AAD_CLIENT_ID,
    AAD_JWKS_PATH,
)

# Override with environment variable if available
SECRET_KEY = os.getenv("JWT_SECRET_KEY", SECRET_KEY)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load Azure AD JWKs if configured
AAD_JWKS = []
if AAD_JWKS_PATH and Path(AAD_JWKS_PATH).exists():
    try:
        AAD_JWKS = json.loads(Path(AAD_JWKS_PATH).read_text()).get("keys", [])
    except Exception:
        AAD_JWKS = []


def get_users_db():
    """Get the users database"""
    users_file = BASE_DIR / "users.json"
    if not users_file.exists():
        # Create default admin user if no users file exists
        default_admin = {
            "admin": {
                "username": "admin",
                "tenant": "*",  # Wildcard for all tenants
                "role": "system_admin",
                "agents": [],
                "hashed_password": pwd_context.hash("admin"),
                "disabled": False,
            }
        }
        users_file.write_text(json.dumps(default_admin, indent=2))
        return default_admin
    return json.loads(users_file.read_text())


def save_users_db(users_data):
    """Save the users database"""
    users_file = BASE_DIR / "users.json"
    users_file.write_text(json.dumps(users_data, indent=2))


def verify_password(plain_password, hashed_password):
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash a password"""
    return pwd_context.hash(password)


def get_user(username: str):
    """Get a user by username"""
    users_db = get_users_db()
    if username in users_db:
        user_data = users_db[username].copy()
        # Remove hashed_password from user data before returning
        if "hashed_password" in user_data:
            user_data.pop("hashed_password")
        return User(**user_data)
    return None


def authenticate_user(username: str, password: str):
    """Authenticate a user"""
    users_db = get_users_db()
    if username not in users_db:
        return False
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, users_db[username]["hashed_password"]):
        return False
    return user


def create_access_token(data: dict):
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_aad_token(token: str):
    """Authenticate a user using an Azure AD token"""
    if not (AAD_TENANT_ID and AAD_CLIENT_ID and AAD_JWKS):
        return None
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        return None
    key = next(
        (k for k in AAD_JWKS if k.get("kid") == unverified_header.get("kid")), None
    )
    if not key:
        return None
    issuer = f"https://login.microsoftonline.com/{AAD_TENANT_ID}/v2.0"
    try:
        payload = jwt.decode(
            token,
            key,
            algorithms=[key.get("alg", "RS256")],
            audience=AAD_CLIENT_ID,
            issuer=issuer,
        )
    except JWTError:
        return None
    username = (
        payload.get("preferred_username") or payload.get("upn") or payload.get("email")
    )
    if not username:
        return None
    user = get_user(username)
    if user:
        return user
    return User(username=username, tenant="*", role="user", agents=[])


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        tenant: str = payload.get("tenant")
        role: str = payload.get("role", "user")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, tenant=tenant, role=role)
    except JWTError:
        raise credentials_exception
    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Get the current active user"""
    users_db = get_users_db()
    if users_db.get(current_user.username, {}).get("disabled", False):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_active_user)):
    """Get the current user if they are an admin"""
    if current_user.role not in ["admin", "system_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


async def get_system_admin_user(current_user: User = Depends(get_current_active_user)):
    """Ensure the user is a system administrator"""
    if current_user.role != "system_admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


def create_user(user_data: UserCreate):
    """Create a new user"""
    users_db = get_users_db()
    if user_data.username in users_db:
        return False

    hashed_password = get_password_hash(user_data.password)
    users_db[user_data.username] = {
        "username": user_data.username,
        "tenant": user_data.tenant,
        "role": user_data.role,
        "agents": user_data.agents or [],
        "disabled": user_data.disabled,
        "hashed_password": hashed_password,
    }
    save_users_db(users_db)
    return True


def update_user(username: str, user_data: dict):
    """Update a user"""
    users_db = get_users_db()
    if username not in users_db:
        return False

    for key, value in user_data.items():
        if key not in ["username", "hashed_password", "password"]:
            users_db[username][key] = value

    if "password" in user_data:
        users_db[username]["hashed_password"] = get_password_hash(user_data["password"])

    save_users_db(users_db)
    return True


def delete_user(username: str):
    """Delete a user"""
    users_db = get_users_db()
    if username not in users_db:
        return False

    del users_db[username]
    save_users_db(users_db)
    return True
