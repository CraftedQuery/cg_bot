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
BASE_UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", BASE_DIR / "uploads"))
DEFAULT_TENANT = "public"
DEFAULT_AGENT = "default"

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


def store_path(tenant: str, agent: str) -> Path:
    """Get vector store path for a tenant/agent"""
    return BASE_STORE_DIR / tenant / agent


def upload_path(tenant: str, agent: str) -> Path:
    """Get path where uploaded files for a tenant/agent are stored"""
    return BASE_UPLOAD_DIR / tenant / agent


def load_config(tenant: str, agent: str) -> Dict[str, Any]:
    """Load configuration for a tenant/agent"""
    p = cfg_path(tenant, agent)
    if p.exists():
        return json.loads(p.read_text())

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
        # Enhanced widget parameters
        "enable_voice": True,
        "enable_files": True,
        "enable_tts": False,
        "enable_dark_mode": True,
        "widget_position": "bottom-right",
        "widget_size": "medium",
        "welcome_message": "Hello! How can I help you today?",
        "placeholder_text": "Type your message...",
    }
    p.write_text(json.dumps(cfg, indent=2))
    return cfg


def save_config(tenant: str, agent: str, cfg: Dict[str, Any]) -> bool:
    """Save configuration for a tenant/agent"""
    p = cfg_path(tenant, agent)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=2))
    return True
