"""
models.py - Pydantic models and schemas for the RAG chatbot
"""

from typing import Any

from pydantic import BaseModel, Field


class User(BaseModel):
    username: str
    tenant: str
    role: str = "user"
    disabled: bool = False
    agents: list[str] = Field(default_factory=list)
    allow_files: bool = False
    language: str = "English"


class UserCreate(User):
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None
    tenant: str | None = None
    role: str | None = None


class AADTokenRequest(BaseModel):
    """Request model for Azure AD login"""

    access_token: str


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    skip_question_evaluation: bool | None = None
    question_evaluation_id: int | None = None
    question_decision: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict[str, Any]]
    question_evaluation: dict[str, Any] | None = None


class ConfigUpdateRequest(BaseModel):
    bot_name: str | None = None
    system_prompt: str | None = None
    primary_color: str | None = None
    secondary_color: str | None = None
    avatar_url: str | None = None
    mode: str | None = None
    auto_open: bool | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    temperature: float | None = None
    allowed_domains: list[str] | None = None
    local_only: bool | None = None


class StageLLMConfig(BaseModel):
    """Configuration for an individual LLM stage."""

    enabled: bool | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    endpoint: str | None = None
    system_prompt: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


class EnhancedConfigUpdateRequest(ConfigUpdateRequest):
    """Enhanced configuration model with widget parameters"""

    enable_voice: bool | None = None
    enable_files: bool | None = None
    enable_tts: bool | None = None
    enable_dark_mode: bool | None = None
    widget_position: str | None = None
    widget_size: str | None = None
    welcome_message: str | None = None
    placeholder_text: str | None = None
    question_evaluator: StageLLMConfig | None = None
    main_rag: StageLLMConfig | None = None
    answer_evaluator: StageLLMConfig | None = None
