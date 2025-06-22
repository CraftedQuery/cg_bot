"""
models.py - Pydantic models and schemas for the RAG chatbot
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class User(BaseModel):
    username: str
    tenant: str
    role: str = "user"
    disabled: bool = False
    agents: Optional[List[str]] = []
    allow_files: bool = False


class UserCreate(User):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    tenant: Optional[str] = None
    role: Optional[str] = None


class AADTokenRequest(BaseModel):
    """Request model for Azure AD login"""

    access_token: str


class ChatRequest(BaseModel):
    messages: List[dict]


class ChatResponse(BaseModel):
    reply: str
    sources: List[dict]


class ConfigUpdateRequest(BaseModel):
    bot_name: Optional[str] = None
    system_prompt: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    avatar_url: Optional[str] = None
    mode: Optional[str] = None
    auto_open: Optional[bool] = None
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: Optional[float] = None
    allowed_domains: Optional[List[str]] = None
    local_only: Optional[bool] = None


class EnhancedConfigUpdateRequest(ConfigUpdateRequest):
    """Enhanced configuration model with widget parameters"""

    enable_voice: Optional[bool] = None
    enable_files: Optional[bool] = None
    enable_tts: Optional[bool] = None
    enable_dark_mode: Optional[bool] = None
    widget_position: Optional[str] = None
    widget_size: Optional[str] = None
    welcome_message: Optional[str] = None
    placeholder_text: Optional[str] = None
