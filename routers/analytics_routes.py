"""
routers/analytics_routes.py - Analytics endpoints
"""
from typing import Optional
from fastapi import APIRouter, Query, Depends, HTTPException

from ..models import User
from ..auth import get_admin_user
from ..config import DEFAULT_TENANT
from ..analytics import get_analytics, get_widget_analytics

router = APIRouter(tags=["analytics"])


@router.get("/analytics")
async def analytics(
    tenant: str = Query(DEFAULT_TENANT),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_admin_user)
):
    """Get analytics for a tenant"""
    
    # Check if user has access to this tenant
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant's analytics"
        )
    
    return get_analytics(tenant, start_date, end_date)


@router.get("/analytics/widget")
async def widget_analytics(
    tenant: str = Query(DEFAULT_TENANT),
    agent: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_admin_user)
):
    """Get widget-specific analytics"""
    
    # Check if user has access to this tenant
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant's analytics"
        )
    
    return get_widget_analytics(tenant, agent, start_date, end_date)