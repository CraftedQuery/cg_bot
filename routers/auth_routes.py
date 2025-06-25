"""
routers/auth_routes.py - Authentication endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ..models import Token, User, UserCreate, AADTokenRequest
from ..auth import (
    authenticate_user,
    authenticate_aad_token,
    create_access_token,
    get_current_active_user,
    get_admin_user,
    get_users_db,
    create_user,
    update_user,
    delete_user,
    get_user,
)

router = APIRouter(tags=["authentication"])


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "tenant": user.tenant, "role": user.role}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/aad/token", response_model=Token)
async def login_with_aad_token(req: AADTokenRequest):
    """Login with Azure AD access token"""
    user = authenticate_aad_token(req.access_token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Azure AD token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username, "tenant": user.tenant, "role": user.role}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_new_user(user: UserCreate, admin: User = Depends(get_admin_user)):
    """Create a new user (admin only)"""
    if admin.role == "system_admin":
        if user.role == "system_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your permissions are not sufficient to complete this action",
            )
    else:
        if user.role != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Your permissions are not sufficient to complete this action",
            )
        user.tenant = admin.tenant
    if create_user(user):
        return {"message": "User created successfully"}
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists"
    )


@router.put("/users/{username}")
async def update_existing_user(
    username: str, user_data: dict, admin: User = Depends(get_admin_user)
):
    """Update an existing user (admin only)"""
    existing = get_user(username)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )
    if admin.role != "system_admin" and existing.tenant != admin.tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your permissions are not sufficient to complete this action",
        )
    if (
        "tenant" in user_data
        and admin.role != "system_admin"
        and user_data["tenant"] != admin.tenant
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your permissions are not sufficient to complete this action",
        )
    if update_user(username, user_data):
        return {"message": "User updated successfully"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.delete("/users/{username}")
async def delete_existing_user(username: str, admin: User = Depends(get_admin_user)):
    """Delete a user (admin only)"""
    if delete_user(username):
        return {"message": "User deleted successfully"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")


@router.get("/users")
async def list_users(admin: User = Depends(get_admin_user)):
    """List all users (admin only)"""
    users_db = get_users_db()
    # Remove sensitive information
    users = []
    for username, user_data in users_db.items():
        if admin.role != "system_admin" and user_data.get("tenant") != admin.tenant:
            continue
        user_info = user_data.copy()
        if "hashed_password" in user_info:
            user_info.pop("hashed_password")
        users.append(user_info)
    return users
