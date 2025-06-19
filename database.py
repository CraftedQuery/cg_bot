"""
database.py - Database operations for the RAG chatbot
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH


def init_database():
    """Initialize the database with required tables"""
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                tenant TEXT,
                agent TEXT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                sources TEXT,
                latency REAL,
                tokens_in INTEGER,
                tokens_out INTEGER,
                user_feedback INTEGER,
                user_ip TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS llm_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                provider TEXT,
                status TEXT,
                error_message TEXT
            )
        """)
        con.commit()


@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def log_chat(
    tenant: str,
    agent: str,
    session_id: str,
    question: str,
    answer: str,
    sources: str,
    latency: float,
    tokens_in: int,
    tokens_out: int,
    user_ip: str
):
    """Log a chat interaction"""
    from datetime import datetime, timezone
    
    with get_db() as con:
        con.execute(
            """INSERT INTO chat_logs
               (ts, tenant, agent, session_id, question, answer, sources, 
                latency, tokens_in, tokens_out, user_ip)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                tenant,
                agent,
                session_id,
                question,
                answer,
                sources,
                latency,
                tokens_in,
                tokens_out,
                user_ip
            )
        )
        con.commit()


def log_llm_event(provider: str, status: str, error_message: str | None = None):
    """Log an LLM request or error"""
    from datetime import datetime, timezone

    with get_db() as con:
        con.execute(
            """INSERT INTO llm_logs
               (ts, provider, status, error_message)
               VALUES (?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                provider,
                status,
                error_message,
            )
        )
        con.commit()


def update_feedback(chat_id: int, feedback: int):
    """Update feedback for a chat interaction"""
    with get_db() as con:
        result = con.execute(
            "UPDATE chat_logs SET user_feedback = ? WHERE id = ?",
            (feedback, chat_id)
        )
        con.commit()
        return result.rowcount > 0


def get_chat_stats(tenant: str, agent: str = None):
    """Get chat statistics for a tenant"""
    with get_db() as con:
        if agent:
            cursor = con.execute(
                """SELECT COUNT(*) as total_chats, 
                          COUNT(DISTINCT session_id) as unique_sessions 
                   FROM chat_logs 
                   WHERE tenant = ? AND agent = ?""",
                (tenant, agent)
            )
        else:
            cursor = con.execute(
                """SELECT COUNT(*) as total_chats, 
                          COUNT(DISTINCT session_id) as unique_sessions 
                   FROM chat_logs 
                   WHERE tenant = ?""",
                (tenant,)
            )
        return cursor.fetchone()