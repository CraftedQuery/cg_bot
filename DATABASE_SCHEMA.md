# Database Schema

The chatbot stores operational data in a lightweight SQLite database (`chat_logs.db`). The path is controlled by `RAG_CHATBOT_HOME` (defaults to the repository root) in `config.py`.

## `chat_logs`

Records every chat turn for analytics and auditing.

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

- `ts` – UTC timestamp when the exchange was logged.
- `tenant` / `agent` – Context that scopes the request.
- `session_id` – Client-provided session key (also used by the widget).
- `question` / `answer` – User prompt and model response.
- `sources` – JSON-encoded citations returned from the vector search.
- `latency`, `tokens_in`, `tokens_out` – Basic performance metrics.
- `user_feedback` – Optional numeric rating submitted later via `/feedback/{chat_id}`.
- `user_ip` – IP address captured for auditing.

The `log_chat` helper in `database.py` inserts rows into this table.

## `llm_logs`

Captures both completion and embedding calls with provider metadata and any error details.

```sql
CREATE TABLE IF NOT EXISTS llm_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    provider TEXT,
    status TEXT,
    tenant TEXT,
    agent TEXT,
    description TEXT,
    error_message TEXT
);
```

The `model` column is added during upgrades via `_ensure_column`. `log_llm_event` writes to this table from `llm.py` and `embedding.py`.

## `uploaded_files`

Tracks every uploaded document and whether OCR or template processing was applied.

```sql
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
);
```

- `size` – File size in bytes.
- `status` – Ingestion lifecycle marker (for example, `in progress` or `ready`).
- `ocr_used` – Boolean flag indicating if OCR was needed during processing.
- `template` – Marks template snippets that can be injected into answers.

Functions such as `add_uploaded_file`, `set_file_ocr_used`, and `set_file_template` update these columns.

## `error_logs`

Stores application errors with contextual routing data.

```sql
CREATE TABLE IF NOT EXISTS error_logs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    endpoint TEXT,
    tenant TEXT,
    agent TEXT,
    message TEXT
);
```

`log_error` is called by the global exception handlers in `main.py` to record trace details.

## Relationships and usage

- Tables share `tenant` and `agent` columns so analytics can be scoped per customer/agent.
- Upload records map to vector stores on disk at `vector_store/<tenant>/<agent>/`.
- Feedback is appended to existing `chat_logs` rows without creating new entries.

The schema favors simplicity and easy portability; migrations are handled by ensuring missing columns exist on startup.
