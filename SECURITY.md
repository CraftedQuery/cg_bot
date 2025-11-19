# Security Features

This document summarizes how the multi-tenant RAG chatbot protects access and data. Use it as a reference when preparing reviews, audits, or cloud hardening work.

## Authentication and authorization

- **JWT access tokens** – `/token` issues short-lived JWTs signed with `JWT_SECRET_KEY`. Secret management can be delegated to the environment (`JWT_SECRET_KEY`).
- **Password hashing** – Credentials in `users.json` are stored with bcrypt; plaintext passwords never persist.
- **Role hierarchy** – `system_admin` (all tenants), `admin` (single tenant), and `user` (restricted agents). Router guards enforce tenant/agent access checks.
- **Azure AD support** – If `AAD_TENANT_ID`, `AAD_CLIENT_ID`, and `AAD_JWKS_PATH` are set, Azure-issued JWTs are validated and mapped to users via `authenticate_aad_token`.

## Configuration and data protection

- **Tenant isolation** – Configs, vector stores, uploads, and logs are namespaced by tenant and agent. Deleting a tenant/agent clears both disk artifacts and database rows.
- **Widget restrictions** – Each agent config includes `allowed_domains` to limit embedding origins.
- **File handling** – Uploaded files are tracked in the `uploaded_files` table with optional OCR markers; template files can be isolated from normal context.
- **Secrets** – API keys and Azure settings are read from environment variables instead of source control.
- **Logging** – `chat_logs`, `llm_logs`, and `error_logs` record IPs, model metadata, and errors for forensics. The CLI dashboard surfaces summaries without exposing sensitive content.

## Network and runtime guidance

- Serve the API behind HTTPS and use TLS termination or a reverse proxy in production.
- Scope provider API keys (OpenAI, Anthropic, Vertex AI) to the minimum permissions required.
- Set `RAG_CHATBOT_HOME` to a secure, backed-up location; rotate and protect `users.json` and `chat_logs.db` backups.

## Hardening checklist

- [ ] Rotate the default `admin/admin` credentials immediately.
- [ ] Provide a strong `JWT_SECRET_KEY` in the runtime environment.
- [ ] Configure `allowed_domains` for every agent before exposing the widget publicly.
- [ ] Limit filesystem permissions on `configs/`, `uploads/`, `vector_store/`, and `chat_logs.db`.
- [ ] Enable Azure AD validation where SSO is required.
