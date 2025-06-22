"""
database.py - Database operations for the RAG chatbot
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH


def _ensure_column(con: sqlite3.Connection, table: str, column_def: str) -> None:
    """Add column to table if it doesn't exist."""
    col_name = column_def.split()[0]
    cur = con.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    if col_name not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")


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
                tenant TEXT,
                agent TEXT,
                description TEXT,
                error_message TEXT
            )
        """)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS uploaded_files(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant TEXT,
                agent TEXT,
                filename TEXT,
                size INTEGER,
                uploaded_at TEXT,
                status TEXT
            )
        """
        )
        # Ensure newer columns exist when upgrading from older versions
        _ensure_column(con, "llm_logs", "tenant TEXT")
        _ensure_column(con, "llm_logs", "agent TEXT")
        _ensure_column(con, "llm_logs", "description TEXT")
        _ensure_column(con, "llm_logs", "error_message TEXT")
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


def log_llm_event(
    provider: str,
    status: str,
    error_message: str | None = None,
    *,
    tenant: str | None = None,
    agent: str | None = None,
    description: str | None = None,
):
    """Log an LLM request or error with optional context"""
    from datetime import datetime, timezone

    with get_db() as con:
        con.execute(
            """INSERT INTO llm_logs
               (ts, provider, status, tenant, agent, description, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                provider,
                status,
                tenant,
                agent,
                description,
                error_message,
            )
        )
        con.commit()


def record_file_upload(tenant: str, agent: str, filename: str, size: int) -> int:
    """Insert a new uploaded file record and return its ID"""
    from datetime import datetime, timezone

    with get_db() as con:
        cur = con.execute(
            """INSERT INTO uploaded_files
               (tenant, agent, filename, size, uploaded_at, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                tenant,
                agent,
                filename,
                size,
                datetime.now(timezone.utc).isoformat(),
                "in progress",
            ),
        )
        con.commit()
        return cur.lastrowid


def update_file_status(file_id: int, status: str) -> None:
    """Update status for an uploaded file"""
    with get_db() as con:
        con.execute(
            "UPDATE uploaded_files SET status = ?, uploaded_at = uploaded_at WHERE id = ?",
            (status, file_id),
        )
        con.commit()


def list_uploaded_files(tenant: str, agent: str):
    """List files for a tenant/agent ordered by upload time desc"""
    with get_db() as con:
        cur = con.execute(
            "SELECT id, filename, size, uploaded_at, status FROM uploaded_files WHERE tenant = ? AND agent = ? ORDER BY uploaded_at DESC",
            (tenant, agent),
        )
        return cur.fetchall()


def delete_uploaded_file(file_id: int):
    """Remove file record"""
    with get_db() as con:
        con.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
        con.commit()


def delete_uploaded_file_by_name(tenant: str, agent: str, filename: str) -> None:
    """Remove file record matching tenant, agent and filename"""
    with get_db() as con:
        con.execute(
            "DELETE FROM uploaded_files WHERE tenant = ? AND agent = ? AND filename = ?",
            (tenant, agent, filename),
        )
        con.commit()


def get_uploaded_file(file_id: int):
    """Get metadata for a single uploaded file"""
    with get_db() as con:
        cur = con.execute(
            "SELECT tenant, agent, filename FROM uploaded_files WHERE id = ?",
            (file_id,),
        )
        return cur.fetchone()


def count_uploaded_files(tenant: str, agent: str | None = None) -> int:
    """Return number of uploaded files for a tenant or specific agent."""
    with get_db() as con:
        if agent is not None:
            cur = con.execute(
                "SELECT COUNT(*) FROM uploaded_files WHERE tenant = ? AND agent = ?",
                (tenant, agent),
            )
        else:
            cur = con.execute(
                "SELECT COUNT(*) FROM uploaded_files WHERE tenant = ?",
                (tenant,),
            )
        return cur.fetchone()[0]


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