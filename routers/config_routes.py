"""
routers/config_routes.py - Configuration management endpoints
"""
from urllib.parse import urlparse
from fastapi import APIRouter, Request, Query, Depends, HTTPException

from ..models import EnhancedConfigUpdateRequest, User
from ..auth import get_current_active_user, get_admin_user, get_system_admin_user
from ..config import (
    DEFAULT_TENANT,
    DEFAULT_AGENT,
    BASE_CONFIG_DIR,
    load_config,
    save_config,
    store_path
)

router = APIRouter(tags=["configuration"])


@router.post("/tenants/{tenant}")
async def create_tenant(
    tenant: str,
    current_user: User = Depends(get_system_admin_user)
):
    """Create a new tenant (system admin only)"""
    tenant_dir = BASE_CONFIG_DIR / tenant
    if tenant_dir.exists():
        raise HTTPException(status_code=400, detail="Tenant already exists")
    tenant_dir.mkdir(parents=True, exist_ok=True)
    return {"message": "Tenant created", "tenant": tenant}


@router.post("/agents/{tenant}/{agent}")
async def create_agent(
    tenant: str,
    agent: str,
    current_user: User = Depends(get_admin_user)
):
    """Create a new agent for a tenant"""
    if current_user.role != "system_admin" and current_user.tenant != tenant:
        raise HTTPException(status_code=403, detail="You don't have access to this tenant")
    load_config(tenant, agent)
    return {"message": "Agent created", "tenant": tenant, "agent": agent}


@router.get("/config")
async def get_config(
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    request: Request = None
):
    """Get public configuration for a tenant/agent"""
    cfg = load_config(tenant, agent)
    
    # Check if domain is allowed (for widget embedding)
    if request and "Origin" in request.headers:
        origin = request.headers["Origin"]
        allowed_domains = cfg.get("allowed_domains", ["*"])
        
        if "*" not in allowed_domains:
            domain = urlparse(origin).netloc
            if domain not in allowed_domains:
                raise HTTPException(
                    status_code=403,
                    detail="Domain not allowed to access this configuration"
                )
    
    # Remove sensitive or internal configuration
    public_cfg = {k: v for k, v in cfg.items() if k not in [
        "llm_provider", "llm_model", "temperature", "allowed_domains"
    ]}
    
    return public_cfg


@router.get("/config/full")
async def get_full_config(
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_current_active_user)
):
    """Get full configuration including all widget parameters"""
    
    # Check if user has access to this tenant
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    cfg = load_config(tenant, agent)
    return cfg


@router.put("/config")
async def update_config_enhanced(
    config: EnhancedConfigUpdateRequest,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_admin_user)
):
    """Update configuration for a tenant/agent"""
    cfg = load_config(tenant, agent)
    
    # Update only provided fields
    for field, value in config.dict(exclude_unset=True).items():
        if value is not None:
            cfg[field] = value
    
    save_config(tenant, agent, cfg)
    return {"message": "Configuration updated successfully", "config": cfg}


@router.get("/tenants")
async def list_tenants(current_user: User = Depends(get_admin_user)):
    """List all tenants and their agents"""
    tenants = []
    for tenant_dir in BASE_CONFIG_DIR.iterdir():
        if tenant_dir.is_dir():
            agents = []
            for config_file in tenant_dir.iterdir():
                if config_file.is_file() and config_file.suffix == ".json":
                    agent_name = config_file.stem
                    agents.append(agent_name)
            tenants.append({"tenant": tenant_dir.name, "agents": agents})
    return tenants


@router.get("/my-agents")
async def get_my_agents(current_user: User = Depends(get_current_active_user)):
    """Get all agents available to the current user's tenant"""
    
    user_tenant = current_user.tenant
    
    # If user has access to all tenants, return all
    if user_tenant == "*":
        tenants = []
        for tenant_dir in BASE_CONFIG_DIR.iterdir():
            if tenant_dir.is_dir():
                agents = []
                for config_file in tenant_dir.iterdir():
                    if config_file.is_file() and config_file.suffix == ".json":
                        agent_name = config_file.stem
                        config = load_config(tenant_dir.name, agent_name)
                        agents.append({
                            "agent": agent_name,
                            "bot_name": config.get("bot_name", f"{tenant_dir.name}-{agent_name}"),
                            "primary_color": config.get("primary_color", "#1E88E5"),
                            "avatar_url": config.get("avatar_url", "")
                        })
                tenants.append({"tenant": tenant_dir.name, "agents": agents})
        return tenants
    
    # Return only agents for the user's specific tenant
    tenant_dir = BASE_CONFIG_DIR / user_tenant
    if not tenant_dir.exists():
        return {"tenant": user_tenant, "agents": []}
    
    agents = []
    allowed = set(current_user.agents or [])
    for config_file in tenant_dir.iterdir():
        if config_file.is_file() and config_file.suffix == ".json":
            agent_name = config_file.stem
            if allowed and agent_name not in allowed:
                continue
            config = load_config(user_tenant, agent_name)

            # Check if vector store exists for this agent
            vector_store_exists = store_path(user_tenant, agent_name).exists()

            agents.append({
                "agent": agent_name,
                "bot_name": config.get("bot_name", f"{user_tenant}-{agent_name}"),
                "primary_color": config.get("primary_color", "#1E88E5"),
                "secondary_color": config.get("secondary_color", "#FFFFFF"),
                "avatar_url": config.get("avatar_url", ""),
                "mode": config.get("mode", "inline"),
                "auto_open": config.get("auto_open", False),
                "vector_store_ready": vector_store_exists
            })
    
    return {
        "tenant": user_tenant,
        "agents": agents,
        "total_agents": len(agents)
    }


@router.get("/agent-status/{tenant}/{agent}")
async def get_agent_status(
    tenant: str,
    agent: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed status of a specific agent"""
    
    # Check if user has access to this tenant
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    # Check if agent exists
    from ..config import cfg_path
    config_file = cfg_path(tenant, agent)
    if not config_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Agent not found"
        )
    
    config = load_config(tenant, agent)
    vector_store_exists = store_path(tenant, agent).exists()
    
    # Get usage statistics from database
    from ..database import get_chat_stats
    total_chats, unique_sessions = get_chat_stats(tenant, agent)
    
    # Get recent activity (last 7 days)
    from datetime import datetime, timedelta, timezone
    from ..database import get_db
    
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    with get_db() as con:
        cursor = con.execute(
            "SELECT COUNT(*) FROM chat_logs WHERE tenant = ? AND agent = ? AND ts > ?",
            (tenant, agent, week_ago)
        )
        recent_chats = cursor.fetchone()[0]
    
    return {
        "tenant": tenant,
        "agent": agent,
        "config": config,
        "vector_store_ready": vector_store_exists,
        "statistics": {
            "total_chats": total_chats,
            "unique_sessions": unique_sessions,
            "recent_chats_7d": recent_chats
        }
    }