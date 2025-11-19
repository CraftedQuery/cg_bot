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

## Logging and analytics

FastAPI bootstraps `logging.basicConfig(level=logging.INFO)` in `main.py`. Application events are also captured in SQLite tables:

- `chat_logs` – prompts, answers, latency, token counts, and per-chat feedback.
- `llm_logs` – provider/model metadata for calls and embedding runs.
- `uploaded_files` – filename, size, OCR flag, and template marker for each upload.
- `error_logs` – exception traces with tenant/agent context.

Use `analytics.py` or the CLI dashboard to summarize usage per tenant or agent.

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
