# Storage Layout

This project keeps its runtime state in a few predictable locations. Paths are resolved relative to `RAG_CHATBOT_HOME` (defaults to the repository root) in `config.py` so you can mount an external volume in production.

## SQLite database

- **File**: `chat_logs.db`
- **Location**: `${RAG_CHATBOT_HOME:-.}/chat_logs.db`
- **Tables**: `chat_logs`, `llm_logs`, `uploaded_files`, and `error_logs` (see `DATABASE_SCHEMA.md`)

## Tenant and agent configuration

- **Directory**: `${RAG_CHATBOT_HOME:-.}/configs/`
- **Layout**: `configs/<tenant>/<agent>.json`
- **Managed by**: `config.py`

Configs are created lazily with defaults (colors, prompts, model settings, widget options) the first time an agent is requested.

### Tenant/agent naming gotchas

Tenant and agent values are used directly as directory/file names (for configs, uploads, and vector stores). Avoid:

- Trailing spaces (problematic on Windows paths)
- Characters your filesystem forbids (for example `:` on Windows)
- Extremely long tenant/agent names that may exceed path-length limits

## Vector stores

- **Directory**: `${RAG_CHATBOT_HOME:-.}/vector_store/`
- **Layout**: `vector_store/<tenant>/<agent>/`
- **Contents**: FAISS index files plus `meta.json` describing the embedding provider/model
- **Managed by**: `vectorstore.py` via `create_vector_store`/`update_vector_store`

If a store is missing, `/chat` returns a 404 prompting you to run ingestion.

## Uploads

- **Directory**: `${RAG_CHATBOT_HOME:-.}/uploads/`
- **Layout**: `uploads/<tenant>/<agent>/`
- **Metadata**: Tracked in the `uploaded_files` table with OCR/template flags

## Users and credentials

- **File**: `${RAG_CHATBOT_HOME:-.}/users.json`
- **Managed by**: `auth.py`

The file is created automatically with a default `admin` user (role `system_admin`) and bcrypt-hashed password. Azure AD settings are loaded from environment variables (`AAD_TENANT_ID`, `AAD_CLIENT_ID`, `AAD_JWKS_PATH`).

## Derived data

- **Logs**: Cumulative records stored in SQLite tables (`chat_logs`, `llm_logs`, `error_logs`).
- **Analytics**: Aggregated at runtime via `analytics.py` or the CLI dashboard; no additional storage is used.

All of these locations are safe to back up or mount externally; clearing a tenant/agent's data involves removing its config, upload directory, vector store, and associated rows in the SQLite tables.
