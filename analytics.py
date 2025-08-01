"""
analytics.py - Analytics and reporting functionality
"""
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import pandas as pd

from pathlib import Path

from .database import get_db, count_uploaded_files
from .vectorstore import get_vector_store_size
from .config import BASE_STORE_DIR


def _total_vector_store_size(tenant: str) -> int:
    """Return total vector store size in bytes for a tenant across all agents."""
    base = BASE_STORE_DIR / tenant
    if not base.exists():
        return 0
    size = 0
    for p in base.rglob("*"):
        if p.is_file():
            size += p.stat().st_size
    return size


def _clean_values(obj: Any) -> Any:
    """Recursively replace NaN/inf values with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _clean_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_values(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def get_analytics(
    tenant: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    agent: Optional[str] = None
) -> Dict[str, Any]:
    """Get analytics for a tenant"""
    
    # Set default date range if not provided (last 7 days)
    if not end_date:
        end_date = datetime.now(timezone.utc).isoformat()
    if not start_date:
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    
    # Build query
    query = "SELECT * FROM chat_logs WHERE tenant = ? AND ts BETWEEN ? AND ?"
    params = [tenant, start_date, end_date]
    
    if agent:
        query += " AND agent = ?"
        params.append(agent)
    
    # Query the database
    with get_db() as con:
        df = pd.read_sql_query(query, con, params=params)
        file_count = count_uploaded_files(tenant, agent) if agent else count_uploaded_files(tenant)

    size_bytes = get_vector_store_size(tenant, agent) if agent else _total_vector_store_size(tenant)
    storage_gb = size_bytes / (1024 ** 3)
    
    # Generate analytics
    if df.empty:
        return {"message": "No data available for selected period"}
    
    return _clean_values({
        "total_queries": len(df),
        "unique_sessions": df['session_id'].nunique(),
        "daily_counts": _get_daily_counts(df),
        "top_questions": _get_top_questions(df),
        "top_sources": _get_top_sources(df),
        "session_stats": _get_session_stats(df),
        "performance": _get_performance_stats(df),
        "token_stats": _get_token_stats(df),
        "feedback": _get_feedback_stats(df),
        "storage_gb": storage_gb,
        "file_count": file_count
    })


def get_widget_analytics(
    tenant: str,
    agent: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Get widget-specific analytics"""
    
    # Set default date range
    if not end_date:
        end_date = datetime.now(timezone.utc).isoformat()
    if not start_date:
        start_date = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    
    # Base query
    query = "SELECT * FROM chat_logs WHERE tenant = ? AND ts BETWEEN ? AND ?"
    params = [tenant, start_date, end_date]
    
    if agent:
        query += " AND agent = ?"
        params.append(agent)
    
    with get_db() as con:
        df = pd.read_sql_query(query, con, params=params)
    
    if df.empty:
        return {"message": "No data available for selected period"}
    
    # Widget-specific analytics
    df['hour'] = pd.to_datetime(df['ts']).dt.hour
    
    return _clean_values({
        "total_interactions": len(df),
        "unique_sessions": df['session_id'].nunique(),
        "avg_session_length": df.groupby('session_id').size().mean(),
        "peak_hours": df['hour'].value_counts().to_dict(),
        "response_times": {
            "avg": df['latency'].mean(),
            "p95": df['latency'].quantile(0.95),
            "p99": df['latency'].quantile(0.99)
        },
        "user_satisfaction": {
            "avg_rating": df['user_feedback'].mean() if 'user_feedback' in df else None,
            "total_ratings": df['user_feedback'].count() if 'user_feedback' in df else 0
        }
    })


def _get_daily_counts(df: pd.DataFrame) -> list:
    """Get daily interaction counts"""
    df['date'] = pd.to_datetime(df['ts']).dt.date
    daily_counts = df.groupby(['date']).size().reset_index(name='count')
    return daily_counts.to_dict('records')


def _get_top_questions(df: pd.DataFrame) -> dict:
    """Get top questions"""
    return df['question'].value_counts().head(10).to_dict()


def _get_top_sources(df: pd.DataFrame) -> dict:
    """Get top sources"""
    sources = []
    for src_json in df['sources'].dropna():
        try:
            src_list = json.loads(src_json)
            for src in src_list:
                if 'source' in src:
                    sources.append(src['source'])
        except:
            pass
    return pd.Series(sources).value_counts().head(10).to_dict()


def _get_session_stats(df: pd.DataFrame) -> dict:
    """Get session statistics"""
    session_counts = df.groupby('session_id').size()
    return session_counts.describe().to_dict()


def _get_performance_stats(df: pd.DataFrame) -> dict:
    """Get performance statistics"""
    return df['latency'].describe().to_dict()


def _get_token_stats(df: pd.DataFrame) -> dict:
    """Get token usage statistics"""
    return {
        'input': df['tokens_in'].describe().to_dict(),
        'output': df['tokens_out'].describe().to_dict()
    }


def _get_feedback_stats(df: pd.DataFrame) -> dict:
    """Get feedback statistics"""
    feedback_counts = df['user_feedback'].value_counts().to_dict()
    avg_feedback = df['user_feedback'].mean() if 'user_feedback' in df else None
    
    return {
        "counts": feedback_counts,
        "average": avg_feedback
    }