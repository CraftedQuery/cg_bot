"""
config.py - Configuration management for the RAG chatbot
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# Determine base data directory
BASE_DIR = Path(os.getenv("RAG_CHATBOT_HOME", Path.cwd()))

# Base configuration
BASE_CONFIG_DIR = BASE_DIR / "configs"
BASE_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
BASE_STORE_DIR = BASE_DIR / "vector_store"
BASE_UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_TENANT = "public"
DEFAULT_AGENT = "default"

# Default question evaluator prompt to keep the stage strictly focused on validation
DEFAULT_QUESTION_EVALUATOR_PROMPT = """
You are ONLY evaluating if a user question is appropriate for the municipal government chatbot. You are NOT answering the question.

Your job is to assess the question against the evaluation criteria and return a brief evaluation summary. NEVER provide information that answers the user's question.

Evaluate based on:
- Is it within scope (city services, policies, procedures, public information)?
- Does it request restricted information (confidential, PII, privileged)?
- Does it ask for services outside our authority (legal advice, medical advice, official decisions)?
- Is it clear and specific enough to answer?

Respond ONLY with JSON in one of these formats:

Pass:
{
  "status": "pass",
  "proceed": true,
  "evaluation_summary": "Question is appropriate and within scope. Asks about [topic], which is permitted. No violations detected.",
  "criteria_met": ["within_scope", "non_confidential", "appropriate_tone", "sufficiently_clear"]
}

Reject:
{
  "status": "reject",
  "proceed": false,
  "evaluation_summary": "Question requests confidential information about personnel matters, which is outside permitted scope.",
  "criteria_failed": ["requests_restricted_info"],
  "user_message": "I cannot answer questions about confidential personnel matters. Please contact HR directly for this information."
}

Suggest:
{
  "status": "suggest",
  "proceed": false,
  "evaluation_summary": "Question is vague and could yield better results if refined.",
  "original_question": "What does the city do?",
  "suggested_question": "What services does the City of Stockton provide to residents?",
  "reason": "More specific question will help retrieve more relevant information."
}
"""

# Database path
DB_PATH = BASE_DIR / "chat_logs.db"

# JWT Configuration
SECRET_KEY = "dev_secret_key_change_in_production"  # Override with env var
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Microsoft Entra ID (Azure AD) Configuration
AAD_TENANT_ID = os.getenv("AAD_TENANT_ID", "")
AAD_CLIENT_ID = os.getenv("AAD_CLIENT_ID", "")
AAD_JWKS_PATH = os.getenv("AAD_JWKS_PATH", "")


def cfg_path(tenant: str, agent: str) -> Path:
    """Get configuration file path for a tenant/agent"""
    return BASE_CONFIG_DIR / tenant / f"{agent}.json"


def _default_stage_config(
    *,
    enabled: bool = False,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str = "",
    endpoint: str = "",
    system_prompt: str = "",
    max_tokens: int | None = 500,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Return a default stage configuration dictionary."""

    return {
        "enabled": enabled,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "endpoint": endpoint,
        "system_prompt": system_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def _apply_stage_defaults(stage_cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge defaults into a stage configuration without overwriting existing values."""

    for key, value in defaults.items():
        stage_cfg.setdefault(key, value)
    return stage_cfg


def _ensure_stage_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure new stage configuration blocks exist for backward compatibility."""

    base_provider = cfg.get("llm_provider", "openai")
    base_model = cfg.get("llm_model", "gpt-4o-mini")
    base_temp = cfg.get("temperature", 0.3)
    system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")

    cfg["main_rag"] = _apply_stage_defaults(
        cfg.get("main_rag", {}),
        _default_stage_config(
            enabled=True,
            provider=base_provider,
            model=base_model,
            temperature=base_temp,
            system_prompt=system_prompt,
        ),
    )
    cfg["question_evaluator"] = _apply_stage_defaults(
        cfg.get("question_evaluator", {}),
        _default_stage_config(
            enabled=False,
            system_prompt=DEFAULT_QUESTION_EVALUATOR_PROMPT,
            model=base_model,
            provider=base_provider,
            temperature=0,
        ),
    )
    cfg["answer_evaluator"] = _apply_stage_defaults(
        cfg.get("answer_evaluator", {}),
        _default_stage_config(enabled=False),
    )

    # Retrieval defaults (MMR tuned for legal RAG)
    cfg.setdefault(
        "retrieval",
        {
            "mode": "mmr",
            "k": 8,
            "fetch_k": 50,
            "lambda_mult": 0.6,
        },
    )

    # HyDE defaults (Claude 3.5 Sonnet)
    cfg.setdefault(
        "hyde",
        {
            "enabled": True,
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.2,
            "max_tokens": 400,
        },
    )
    return cfg


def store_path(tenant: str, agent: str) -> Path:
    """Get vector store path for a tenant/agent"""
    return BASE_STORE_DIR / tenant / agent


def uploads_path(tenant: str, agent: str) -> Path:
    """Get uploads path for a tenant/agent"""
    return BASE_UPLOAD_DIR / tenant / agent


def load_config(tenant: str, agent: str) -> Dict[str, Any]:
    """Load configuration for a tenant/agent"""
    p = cfg_path(tenant, agent)
    if p.exists():
        cfg = json.loads(p.read_text())
        if "local_only" not in cfg:
            cfg["local_only"] = True
        cfg = _ensure_stage_defaults(cfg)
        p.write_text(json.dumps(cfg, indent=2))
        return cfg

    # Create default configuration
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "bot_name": f"{tenant}-{agent}-Bot",
        "system_prompt": "You are a helpful assistant.",
        "primary_color": "#1E88E5",
        "secondary_color": "#FFFFFF",
        "avatar_url": "",
        "mode": "inline",
        "auto_open": False,
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "temperature": 0.3,
        "allowed_domains": ["*"],
        "local_only": True,
        # Enhanced widget parameters
        "enable_voice": True,
        "enable_files": True,
        "enable_tts": False,
        "enable_dark_mode": True,
        "widget_position": "bottom-right",
        "widget_size": "medium",
        "welcome_message": "Hello! How can I help you today?",
        # Placeholder shown in the widget's input box
        "placeholder_text": "Please ask your question...",
        "question_evaluator": _default_stage_config(
            enabled=False,
            provider="openai",
            model="gpt-4o-mini",
            temperature=0,
            system_prompt=DEFAULT_QUESTION_EVALUATOR_PROMPT,
        ),
        "main_rag": _default_stage_config(
            enabled=True,
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.3,
            system_prompt="You are a helpful assistant.",
        ),
        "answer_evaluator": _default_stage_config(enabled=False),
        "retrieval": {
            "mode": "mmr",
            "k": 8,
            "fetch_k": 50,
            "lambda_mult": 0.6,
        },
        "hyde": {
            "enabled": True,
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.2,
            "max_tokens": 400,
        },
    }
    p.write_text(json.dumps(cfg, indent=2))
    return cfg


def save_config(tenant: str, agent: str, cfg: Dict[str, Any]) -> bool:
    """Save configuration for a tenant/agent"""
    p = cfg_path(tenant, agent)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    return True
