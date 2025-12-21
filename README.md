# Multi-tenant RAG Chatbot

This repository contains a production-focused Retrieval-Augmented Generation (RAG) chatbot. It supports multiple tenants and agents, protects access with JWTs, offers optional Microsoft Entra (Azure AD) support, and ships with a customizable web widget and CLI utilities.

## Key features

- 🏢 **Multi-tenant RAG** – Separate configs, vector stores, uploads, and analytics for every tenant/agent pair.
- 🔐 **Authentication** – Username/password JWTs with hashed credentials plus optional Azure AD token validation.
- 👑 **Roles** – `system_admin`, `admin`, and `user` roles with tenant scoping and agent assignment.
- 🤖 **Multiple LLMs** – OpenAI by default, with Anthropic and Vertex AI hooks for chat.
- 🧠 **Embeddings** – Pluggable embedding providers (OpenAI, Anthropic, Vertex AI, or local Hugging Face) persisted in FAISS.
- 📂 **Flexible ingestion** – Process local files, Google Drive folders, or sitemap URLs; OCR usage is tracked automatically.
- 🎛️ **Admin utilities** – Rich-powered CLI dashboard, user management, and ingestion helpers.
- 🧩 **Embeddable widget** – Configurable chat widget with dark mode, voice input, file attachments, and source citations.
- 📊 **Analytics** – Chat, LLM, file, and error logs stored in SQLite for reporting.

## Repository layout

```
cg_bot/
├── main.py              # FastAPI application factory and router wiring
├── cli.py               # Rich-based CLI (serve, dashboard, ingest, user management)
├── auth.py              # JWT + Azure AD auth flows and user store
├── config.py            # Paths, defaults, and tenant/agent configuration helpers
├── database.py          # SQLite schema and logging helpers
├── llm.py               # LLM provider selection (OpenAI, Anthropic, Vertex AI)
├── embedding.py         # Embedding model helpers and logging
├── vectorstore.py       # Chunking utilities and FAISS persistence
├── ingestion.py         # Ingestion pipeline for files, Drive, and sitemaps
├── widget.py            # Chat widget JavaScript generator
├── analytics.py         # Basic usage analytics helpers
├── routers/             # FastAPI routers (auth, chat, config, admin, analytics, ingest, users)
├── utils/               # Google Drive, sitemap scraping, and file processing helpers
├── static/              # Admin HTML and other static assets served by FastAPI
├── docs/                # Supplementary documentation (Entra, background jobs)
├── configs/             # Runtime tenant/agent configs (created on demand)
├── vector_store/        # FAISS indexes (created on demand)
├── uploads/             # Uploaded files tracked per tenant/agent
└── users.json           # User store (created on first run)
```

See `project-structure.txt` for a more detailed walkthrough of the codebase.

## Prerequisites

- Python 3.11+
- Optional: Node 18+ if you plan to run the sample SPA in `spa/`
- API keys for the LLM and embedding providers you plan to use (OpenAI, Anthropic, Vertex AI)

## Setup

```bash
git clone <repository-url>
cd cg_bot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required for OpenAI chat + embeddings |
| `ANTHROPIC_API_KEY` | Required when using Anthropic chat/embeddings |
| `GOOGLE_APPLICATION_CREDENTIALS` | Service account JSON for Drive + Vertex AI |
| `JWT_SECRET_KEY` | Overrides the default JWT signing secret |
| `RAG_CHATBOT_HOME` | Root directory for configs, uploads, vector stores, and the SQLite DB |
| `AAD_TENANT_ID`, `AAD_CLIENT_ID`, `AAD_JWKS_PATH` | Enable optional Azure AD JWT validation |

Unset variables simply disable the corresponding provider or feature.

## Running the API

Start the FastAPI server with the CLI (from the repository root):

```bash
python cli.py serve --reload
```

The CLI sets up module paths automatically; no packaging step is required. Once running:

- API docs: http://localhost:8000/docs
- Admin page: http://localhost:8000/admin.html
- Widget script: http://localhost:8000/widget.js?tenant=public&agent=default

### Default credentials

- Username: `admin`
- Password: `admin`
- Role: `system_admin`

Change these immediately in production—`auth.py` hashes passwords with bcrypt and stores them in `users.json`.

## CLI quick reference

```bash
# Launch the Rich dashboard
python cli.py dashboard

# Start the API server
python cli.py serve --host 0.0.0.0 --port 8000

# Ingest content
python cli.py ingest TENANT AGENT --file ./docs/guide.pdf --sitemap https://example.com/sitemap.xml

# Create a user
python cli.py create-user alice S3cret! --tenant public --role admin --agents default
```

The dashboard view summarizes tenants, vector store presence, users, and ingestion helpers without writing code.

## API usage

### Obtain a token

```python
import requests

resp = requests.post("http://localhost:8000/token", data={"username": "admin", "password": "admin"})
token = resp.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}
```

### Send a chat request

```python
payload = {
    "messages": [
        {"role": "user", "content": "Give me a 2 sentence summary of the refund policy."}
    ]
}

r = requests.post(
    "http://localhost:8000/chat",
    params={"tenant": "public", "agent": "default"},
    headers=headers,
    json=payload,
)
print(r.json())  # {"reply": "...", "sources": [{"source": "..."}]}
```

### Widget embedding

Drop the widget loader onto any page and point it at the tenant/agent you want to expose:

```html
<script src="http://your-server.com/widget.js?tenant=public&agent=default"></script>
```

The widget honors each agent's config file (colors, features, welcome text, etc.) and injects citations using the FAISS search results.

## Configurations, data, and storage

- **Configs** – Stored at `configs/<tenant>/<agent>.json`. Created lazily with sensible defaults via `config.py`.
- **Vector stores** – FAISS indexes under `vector_store/<tenant>/<agent>/` with `meta.json` describing the embedding provider/model.
- **Uploads** – Raw uploads live in `uploads/<tenant>/<agent>/` and are tracked in the `uploaded_files` database table.
- **Database** – `chat_logs.db` (or `RAG_CHATBOT_HOME/chat_logs.db`) holds chat, LLM, upload, and error logs. See `DATABASE_SCHEMA.md` for details.
- **Users** – `users.json` contains hashed credentials, tenants, roles, and agent assignments.

Set `RAG_CHATBOT_HOME` to an external volume in production so configs, uploads, and the SQLite database are persisted.

## Ingestion workflow

`ingestion.py` consolidates three sources:

- **Local files** – PDFs, DOCX, text, and more processed through `utils/file_processors.py` (with OCR when needed).
- **Google Drive** – Folder ingestion via `utils/google_drive.py`.
- **Sitemaps** – URL harvesting with `utils/web_scraper.py`.

`update_vector_store` chunks text, builds embeddings with the selected provider, logs embedding events, and saves to FAISS. The ingestion function returns an OCR usage map so callers can record which files needed OCR.

## Three-stage chat pipeline

Every agent configuration includes a staged pipeline that can be tuned independently per tenant/agent pair:

1. **Question evaluator (optional)** – Pre-processes the latest user message before retrieval. You can turn it on or off per agent, pick a provider (OpenAI, Anthropic, Vertex AI, or a custom endpoint), supply a dedicated API key/URL, and set model, temperature, token cap, and a stage-specific system prompt. Results are logged through `log_question_evaluation` so you can review how the guardrail behaved during a conversation.
2. **Main RAG bot** – Always present and enabled by default. This stage gathers vector search results for the tenant/agent, separates template files from normal content, detects the user's language, and builds a system prompt that enforces local-only answers when `local_only` is set. It then calls the selected model/provider with the configured temperature and token cap, and logs the interaction (latency, token counts, feedback hooks, and citations) via `log_chat`.
3. **Answer evaluator (optional)** – Runs after a reply is generated. Like the question evaluator, it supports its own provider/model/API key, prompt, and generation settings. Its verdict and telemetry are captured through `log_answer_evaluation`, and the chat log links back to both evaluation stages for auditing.

Stage defaults live in each agent's config file (`configs/<tenant>/<agent>.json`), and `config.py` backfills missing blocks for older configs. The admin UI exposes toggles, provider/model pickers, API-key overrides, and readiness badges for each stage so you can validate the full pipeline before saving.

**📖 Configuring System Prompts:** All system prompts for each LLM call are fully configurable through the Admin UI. See [`docs/system-prompts-configuration.md`](docs/system-prompts-configuration.md) for a complete guide on where and how to configure the context for each stage (Question Evaluator, Main RAG, HyDE, JSON Repair, and Answer Evaluator).

## RAG request flow and provider calls

Even when you “only” configure a question evaluator and a main RAG model, a RAG request can legitimately trigger **multiple external calls**:

- **Question evaluator (optional LLM call)**: Runs first when `question_evaluator.enabled` is true.
- **HyDE (optional LLM call)**: When `hyde.enabled` is true, the system generates a *hypothetical* excerpt to improve retrieval. This is a separate LLM call and is **not** MMR.
- **Embeddings (embedding model call)**: Retrieval against FAISS requires embedding the query (HyDE output or the original question). This happens for both `retrieval.mode="mmr"` and `"similarity"`.
- **Answer generation (LLM call)**: The main RAG stage generates a structured JSON answer anchored to retrieved evidence.
- **JSON repair (optional extra LLM call)**: If the answer is not valid JSON, the pipeline performs **one** repair attempt.
- **Answer evaluator (optional LLM call)**: Runs last when `answer_evaluator.enabled` is true.

Providers/models for each stage are configurable per tenant/agent (see `configs/<tenant>/<agent>.json`).

For a detailed, step-by-step sequence (including HyDE vs MMR and why embeddings show up), see [`docs/rag-request-flow.md`](docs/rag-request-flow.md).

## Logging and analytics

FastAPI bootstraps `logging.basicConfig(level=logging.INFO)` in `main.py`. Application events are also captured in SQLite tables:

- `chat_logs` – prompts, answers, latency, token counts, and per-chat feedback.
- `llm_logs` – provider/model metadata for calls and embedding runs.
- `uploaded_files` – filename, size, OCR flag, and template marker for each upload.
- `error_logs` – exception traces with tenant/agent context.

Use `analytics.py` or the CLI dashboard to summarize usage per tenant or agent.

## Admin console walkthrough

`/admin.html` ships with a built-in login form (JWT-backed) and a tabbed console for day-to-day operations:

- **Dashboard** – Summarizes total tenants, agents, users, and chat volume so you can confirm the instance is populated before drilling into details.
- **Tenants & Agents** – Lists every tenant with its agents and action buttons to:
  - Create a tenant (with an initial agent) via the modal dropdown/text input flow.
  - Add agents to an existing tenant.
  - Configure an agent’s bot, widget appearance, allowed domains, and the full three-stage pipeline (providers, models, prompts, token/temperature caps, API-key overrides, and ready/incomplete badges for each stage).
  - Trigger a test chat, upload files into the agent’s vector store, copy the embeddable widget script, or delete the agent.
- **Users** – Displays a sortable table of usernames, tenant scope, agent assignments, role, file-permission toggle, status, and actions. Modals let you create users (username/password, tenant, role, agents, file permissions) or edit existing users, and non-admin accounts can be deleted when allowed.
- **Analytics** – Prompts for a tenant selection and then surfaces total queries, unique sessions, average feedback, mean response time, vector store size, and file counts for that tenant.
- **Settings** – Shows system health/versions from `/health`, runs a live LLM connectivity test, reveals the most recent LLM calls, and exposes recent API error logs. You can also adjust the inactivity timeout in minutes.
- **Styling** – Lets you adjust the global theme tokens (primary/secondary/danger/warning palette, grayscale steps, text colors, button radius, body font size, header/footer colors, logo URL, and footer links) used across the admin experience.

## Security considerations

- Override `JWT_SECRET_KEY` in production and rotate it periodically.
- Change the default admin password on first boot; `users.json` stores bcrypt hashes only.
- Restrict widget embedding domains in each agent config (`allowed_domains`).
- Serve behind HTTPS and lock down API keys and Azure AD credentials with your secrets manager.
- Audit logs include IP addresses and optional user feedback ratings for monitoring.

## Development tips

- Run the API locally with `python cli.py serve --reload`.
- Execute the test suite with `pytest` from the repo root.
- Add new providers by extending `llm.py` or `embedding.py`, and new file types by extending `utils/file_processors.py`.

## Support

- Issues: open a ticket in this repository.
- Additional guides: see the `docs/` directory for Azure AD setup and background ingestion notes.
