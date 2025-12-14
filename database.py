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
                user_ip TEXT,
                question_evaluation_id INTEGER,
                answer_evaluation_id INTEGER
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
                status TEXT,
                ocr_used INTEGER DEFAULT 0,
                template INTEGER DEFAULT 0
            )
        """
        )
        # Ensure newer columns exist when upgrading from older versions
        _ensure_column(con, "llm_logs", "tenant TEXT")
        _ensure_column(con, "llm_logs", "agent TEXT")
        _ensure_column(con, "llm_logs", "model TEXT")
        _ensure_column(con, "llm_logs", "description TEXT")
        _ensure_column(con, "llm_logs", "error_message TEXT")
        # Enhanced error logging fields
        _ensure_column(con, "llm_logs", "user TEXT")
        _ensure_column(con, "llm_logs", "question TEXT")
        _ensure_column(con, "llm_logs", "stage TEXT")
        _ensure_column(con, "llm_logs", "request_payload TEXT")
        _ensure_column(con, "llm_logs", "response_payload TEXT")
        _ensure_column(con, "llm_logs", "error_type TEXT")
        _ensure_column(con, "llm_logs", "error_details TEXT")
        _ensure_column(con, "llm_logs", "latency_ms REAL")
        _ensure_column(con, "llm_logs", "tokens_in INTEGER")
        _ensure_column(con, "llm_logs", "tokens_out INTEGER")
        _ensure_column(con, "uploaded_files", "ocr_used INTEGER DEFAULT 0")
        _ensure_column(con, "uploaded_files", "template INTEGER DEFAULT 0")
        _ensure_column(con, "chat_logs", "question_evaluation_id INTEGER")
        _ensure_column(con, "chat_logs", "answer_evaluation_id INTEGER")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS question_evaluation_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                tenant TEXT,
                agent TEXT,
                session_id TEXT,
                conversation_id INTEGER,
                original_question TEXT,
                evaluation_result TEXT,
                provider TEXT,
                model TEXT,
                tokens_used INTEGER,
                latency_ms REAL,
                error TEXT
            )
            """
        )
        _ensure_column(con, "question_evaluation_logs", "username TEXT")
        _ensure_column(con, "question_evaluation_logs", "evaluation_details TEXT")
        _ensure_column(con, "question_evaluation_logs", "flags TEXT")
        _ensure_column(con, "question_evaluation_logs", "prompt TEXT")
        _ensure_column(con, "question_evaluation_logs", "full_response TEXT")
        _ensure_column(con, "question_evaluation_logs", "criteria_scores TEXT")
        _ensure_column(con, "question_evaluation_logs", "evaluation_status TEXT")
        _ensure_column(con, "question_evaluation_logs", "reason TEXT")
        _ensure_column(con, "question_evaluation_logs", "suggested_question TEXT")
        _ensure_column(con, "question_evaluation_logs", "user_choice TEXT")
        _ensure_column(con, "question_evaluation_logs", "proceeded INTEGER")
        _ensure_column(con, "question_evaluation_logs", "final_question TEXT")
        _ensure_column(con, "question_evaluation_logs", "proceed_recommendation INTEGER")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS answer_evaluation_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                tenant TEXT,
                agent TEXT,
                session_id TEXT,
                conversation_id INTEGER,
                original_answer TEXT,
                evaluation_result TEXT,
                provider TEXT,
                model TEXT,
                tokens_used INTEGER,
                latency_ms REAL,
                error TEXT
            )
            """
        )
        _ensure_column(con, "answer_evaluation_logs", "username TEXT")
        _ensure_column(con, "answer_evaluation_logs", "evaluation_details TEXT")
        _ensure_column(con, "answer_evaluation_logs", "flags TEXT")
        _ensure_column(con, "answer_evaluation_logs", "issues TEXT")
        _ensure_column(con, "answer_evaluation_logs", "recommendations TEXT")
        _ensure_column(con, "answer_evaluation_logs", "selected_answer_provider TEXT")
        _ensure_column(con, "answer_evaluation_logs", "prompt TEXT")
        _ensure_column(con, "answer_evaluation_logs", "full_response TEXT")
        _ensure_column(con, "answer_evaluation_logs", "criteria_scores TEXT")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS error_logs(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                endpoint TEXT,
                tenant TEXT,
                agent TEXT,
                message TEXT
            )
            """
        )
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
    user_ip: str,
    *,
    question_evaluation_id: int | None = None,
    answer_evaluation_id: int | None = None,
) -> int:
    """Log a chat interaction and return the chat record ID."""
    from datetime import datetime, timezone

    with get_db() as con:
        cur = con.execute(
            """INSERT INTO chat_logs
               (ts, tenant, agent, session_id, question, answer, sources,
                latency, tokens_in, tokens_out, user_ip, question_evaluation_id,
                answer_evaluation_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                user_ip,
                question_evaluation_id,
                answer_evaluation_id,
            )
        )
        con.commit()
        return cur.lastrowid


def log_llm_event(
    provider: str,
    status: str,
    error_message: str | None = None,
    *,
    tenant: str | None = None,
    agent: str | None = None,
    model: str | None = None,
    description: str | None = None,
    user: str | None = None,
    question: str | None = None,
    stage: str | None = None,
    request_payload: str | None = None,
    response_payload: str | None = None,
    error_type: str | None = None,
    error_details: str | None = None,
    latency_ms: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
):
    """Log an LLM request or error with optional context and enhanced error details"""
    from datetime import datetime, timezone
    import json

    # Truncate large payloads to prevent database bloat (keep first 50KB)
    MAX_PAYLOAD_SIZE = 50000
    if request_payload and len(request_payload) > MAX_PAYLOAD_SIZE:
        request_payload = request_payload[:MAX_PAYLOAD_SIZE] + "\n... [truncated]"
    if response_payload and len(response_payload) > MAX_PAYLOAD_SIZE:
        response_payload = response_payload[:MAX_PAYLOAD_SIZE] + "\n... [truncated]"
    if error_details and len(error_details) > MAX_PAYLOAD_SIZE:
        error_details = error_details[:MAX_PAYLOAD_SIZE] + "\n... [truncated]"

    with get_db() as con:
        con.execute(
            """INSERT INTO llm_logs
               (ts, provider, status, tenant, agent, model, description, error_message,
                user, question, stage, request_payload, response_payload, error_type,
                error_details, latency_ms, tokens_in, tokens_out)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                provider,
                status,
                tenant,
                agent,
                model,
                description,
                error_message,
                user,
                question,
                stage,
                request_payload,
                response_payload,
                error_type,
                error_details,
                latency_ms,
                tokens_in,
                tokens_out,
            )
        )
        con.commit()


def log_question_evaluation(
    *,
    tenant: str,
    agent: str,
    session_id: str,
    conversation_id: int | None,
    original_question: str,
    evaluation_result: str,
    provider: str | None,
    model: str | None,
    tokens_used: int | None,
    latency_ms: float | None,
    error: str | None,
    username: str | None = None,
    evaluation_details: str | None = None,
    flags: str | None = None,
    prompt: str | None = None,
    full_response: str | None = None,
    criteria_scores: str | None = None,
    evaluation_status: str | None = None,
    reason: str | None = None,
    suggested_question: str | None = None,
    proceeded: bool | None = None,
    user_choice: str | None = None,
    final_question: str | None = None,
    proceed_recommendation: bool | None = None,
) -> int:
    """Record a question evaluation stage result and return its ID."""
    from datetime import datetime, timezone

    with get_db() as con:
        cur = con.execute(
            """INSERT INTO question_evaluation_logs
               (ts, tenant, agent, session_id, conversation_id, original_question, evaluation_result,
                provider, model, tokens_used, latency_ms, error, username, evaluation_details, flags,
                prompt, full_response, criteria_scores, evaluation_status, reason, suggested_question,
                user_choice, proceeded, final_question, proceed_recommendation)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                tenant,
                agent,
                session_id,
                conversation_id,
                original_question,
                evaluation_result,
                provider,
                model,
                tokens_used,
                latency_ms,
                error,
                username,
                evaluation_details,
                flags,
                prompt,
                full_response,
                criteria_scores,
                evaluation_status,
                reason,
                suggested_question,
                user_choice,
                int(proceeded) if proceeded is not None else None,
                final_question,
                int(proceed_recommendation) if proceed_recommendation is not None else None,
            ),
        )
        con.commit()
        return cur.lastrowid


def update_question_evaluation_decision(
    evaluation_id: int,
    *,
    user_choice: str | None = None,
    proceeded: bool | None = None,
    final_question: str | None = None,
) -> None:
    """Update a question evaluation record with user decision and flow outcome."""

    fields = []
    params: list = []

    if user_choice is not None:
        fields.append("user_choice = ?")
        params.append(user_choice)
    if proceeded is not None:
        fields.append("proceeded = ?")
        params.append(int(proceeded))
    if final_question is not None:
        fields.append("final_question = ?")
        params.append(final_question)

    if not fields:
        return

    params.append(evaluation_id)

    with get_db() as con:
        con.execute(
            f"UPDATE question_evaluation_logs SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        con.commit()


def log_answer_evaluation(
    *,
    tenant: str,
    agent: str,
    session_id: str,
    conversation_id: int | None,
    original_answer: str,
    evaluation_result: str,
    provider: str | None,
    model: str | None,
    tokens_used: int | None,
    latency_ms: float | None,
    error: str | None,
    username: str | None = None,
    evaluation_details: str | None = None,
    flags: str | None = None,
    issues: str | None = None,
    recommendations: str | None = None,
    selected_answer_provider: str | None = None,
    prompt: str | None = None,
    full_response: str | None = None,
    criteria_scores: str | None = None,
) -> int:
    """Record an answer evaluation stage result and return its ID."""
    from datetime import datetime, timezone

    with get_db() as con:
        cur = con.execute(
            """INSERT INTO answer_evaluation_logs
               (ts, tenant, agent, session_id, conversation_id, original_answer, evaluation_result,
                provider, model, tokens_used, latency_ms, error, username, evaluation_details, flags,
                issues, recommendations, selected_answer_provider, prompt, full_response, criteria_scores)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                tenant,
                agent,
                session_id,
                conversation_id,
                original_answer,
                evaluation_result,
                provider,
                model,
                tokens_used,
                latency_ms,
                error,
                username,
                evaluation_details,
                flags,
                issues,
                recommendations,
                selected_answer_provider,
                prompt,
                full_response,
                criteria_scores,
            ),
        )
        con.commit()
        return cur.lastrowid


def link_stage_conversation(
    chat_id: int,
    question_eval_id: int | None,
    answer_eval_id: int | None,
) -> None:
    """Link evaluation logs to the main chat record."""

    with get_db() as con:
        if question_eval_id is not None:
            con.execute(
                "UPDATE question_evaluation_logs SET conversation_id = ? WHERE id = ?",
                (chat_id, question_eval_id),
            )
        if answer_eval_id is not None:
            con.execute(
                "UPDATE answer_evaluation_logs SET conversation_id = ? WHERE id = ?",
                (chat_id, answer_eval_id),
            )
        con.commit()


def log_error(
    endpoint: str,
    message: str,
    *,
    tenant: str | None = None,
    agent: str | None = None,
) -> None:
    """Log an application error with optional context"""
    from datetime import datetime, timezone

    with get_db() as con:
        con.execute(
            """INSERT INTO error_logs
               (ts, endpoint, tenant, agent, message)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                endpoint,
                tenant,
                agent,
                message,
            ),
        )
        con.commit()


def get_error_logs(limit: int = 100):
    """Retrieve recent error logs"""
    with get_db() as con:
        cur = con.execute(
            "SELECT ts, endpoint, tenant, agent, message FROM error_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()


def record_file_upload(
    tenant: str,
    agent: str,
    filename: str,
    size: int,
    *,
    ocr_used: bool = False,
    template: bool = False,
) -> int:
    """Insert a new uploaded file record and return its ID"""
    from datetime import datetime, timezone

    with get_db() as con:
        cur = con.execute(
            """INSERT INTO uploaded_files
               (tenant, agent, filename, size, uploaded_at, status, ocr_used, template)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tenant,
                agent,
                filename,
                size,
                datetime.now(timezone.utc).isoformat(),
                "in progress",
                int(ocr_used),
                int(template),
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


def set_file_ocr_used(file_id: int, used: bool) -> None:
    """Mark a file record as having used OCR"""
    with get_db() as con:
        con.execute(
            "UPDATE uploaded_files SET ocr_used = ? WHERE id = ?",
            (int(used), file_id),
        )
        con.commit()


def set_file_template(file_id: int, template: bool) -> None:
    """Mark a file record as a template or not"""
    with get_db() as con:
        con.execute(
            "UPDATE uploaded_files SET template = ? WHERE id = ?",
            (int(template), file_id),
        )
        con.commit()


def list_uploaded_files(tenant: str, agent: str):
    """List files for a tenant/agent ordered by upload time desc"""
    with get_db() as con:
        cur = con.execute(
            "SELECT id, filename, size, uploaded_at, status, ocr_used, template FROM uploaded_files WHERE tenant = ? AND agent = ? ORDER BY uploaded_at DESC",
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


def delete_agent_data(tenant: str, agent: str) -> None:
    """Delete all records related to an agent."""
    with get_db() as con:
        con.execute(
            "DELETE FROM chat_logs WHERE tenant = ? AND agent = ?",
            (tenant, agent),
        )
        con.execute(
            "DELETE FROM uploaded_files WHERE tenant = ? AND agent = ?",
            (tenant, agent),
        )
        con.execute(
            "DELETE FROM llm_logs WHERE tenant = ? AND agent = ?",
            (tenant, agent),
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


def is_template_file(tenant: str, agent: str, filename: str) -> bool:
    """Return True if the given uploaded file is marked as a template."""
    with get_db() as con:
        cur = con.execute(
            "SELECT template FROM uploaded_files WHERE tenant = ? AND agent = ? AND filename = ?",
            (tenant, agent, filename),
        )
        row = cur.fetchone()
        return bool(row[0]) if row else False


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