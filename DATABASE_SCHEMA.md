# Database Schema

This project uses a small SQLite database for storing chat interactions. The database file is located at `chat_logs.db` and is initialized by `database.py`.

## Table: `chat_logs`

The `chat_logs` table keeps a record of each user interaction with the chatbot. It is created in `database.py` as shown below:
```sql
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
);
```

### Fields
- **id** – Automatically incrementing identifier for each record.
- **ts** – Timestamp (UTC ISO format) when the interaction occurred.
- **tenant** – Tenant name that owns the conversation.
- **agent** – Agent identifier within the tenant.
- **session_id** – Session identifier used to group conversations.
- **question** – User’s question text.
- **answer** – Generated answer text from the LLM.
- **sources** – JSON-encoded list of document sources returned by the RAG search.
- **latency** – Time in seconds taken to generate the response.
- **tokens_in** – Number of tokens sent to the LLM.
- **tokens_out** – Number of tokens returned by the LLM.
- **user_feedback** – Optional numeric rating submitted by the user (1–5).
- **user_ip** – IP address of the user making the request.

### Purpose
The `chat_logs` table serves multiple roles:
1. **Audit Trail** &mdash; Provides a historical record of all chats for a tenant and agent.
2. **Analytics** &mdash; Used by the analytics functions (`analytics.py`) to compute metrics such as total queries, unique sessions, and performance statistics.
3. **Feedback Tracking** &mdash; The `user_feedback` field is updated through the `/feedback/{chat_id}` endpoint to record user satisfaction.

The `database.py` module also provides helper functions to insert new chat records (`log_chat`), update user feedback (`update_feedback`), and obtain basic statistics (`get_chat_stats`).

## Table: `llm_logs`

The `llm_logs` table captures each request to an LLM provider along with any error message.

```sql
CREATE TABLE IF NOT EXISTS llm_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    provider TEXT,
    status TEXT,
    tenant TEXT,
    agent TEXT,
    model TEXT,
    description TEXT,
    error_message TEXT
);
```

Fields:

- **id** – Incrementing identifier for each log entry.
- **ts** – Timestamp when the request was made.
- **provider** – The LLM provider name.
- **status** – Either `success` or `error`.
- **tenant** – Tenant associated with the call.
- **agent** – Agent name.
- **model** – Model name used for the request.
- **description** – Additional context (file name, user question, etc.).
- **error_message** – Error text if the request failed.

