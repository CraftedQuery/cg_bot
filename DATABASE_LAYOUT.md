# Database Tables and Storage Layout

This document summarizes the storage layout used by the multi-tenant RAG Chatbot.
It includes the SQLite schema, locations of configuration files, vector stores
and user records.

## Database

The application logs all chat interactions in a SQLite database at
`chat_logs.db`. The schema is created in `database.py` and contains a single
`chat_logs` table:

| Column        | Type    | Description                                     |
|---------------|---------|-------------------------------------------------|
| `id`          | INTEGER | Primary key                                     |
| `ts`          | TEXT    | UTC timestamp of the interaction                |
| `tenant`      | TEXT    | Tenant identifier                               |
| `agent`       | TEXT    | Agent name within the tenant                    |
| `session_id`  | TEXT    | Session identifier                              |
| `question`    | TEXT    | User question                                   |
| `answer`      | TEXT    | Assistant reply                                 |
| `sources`     | TEXT    | JSON encoded citation list                      |
| `latency`     | REAL    | Response latency in seconds                     |
| `tokens_in`   | INTEGER | Number of input tokens                          |
| `tokens_out`  | INTEGER | Number of output tokens                         |
| `user_feedback` | INTEGER | Optional thumbs up/down feedback              |
| `user_ip`     | TEXT    | Request IP address for auditing                 |

## Configuration Files

Each tenant/agent pair stores its configuration as JSON under
`configs/<tenant>/<agent>.json`. When a configuration file does not exist,
`config.py` generates one with default values. Example settings include
`bot_name`, `system_prompt`, colour values, LLM provider/model and widget
options. These files are created at runtime so the `configs/` directory may be
empty when the application is first installed.

## Vector Stores

Document embeddings for every tenant/agent live under
`vector_store/<tenant>/<agent>/`. The vector stores are built using FAISS and
are loaded via `vectorstore.py`. If the directory for a tenant/agent does not
exist an HTTP 404 error is raised until ingestion is performed.

## Users

User accounts are stored in the JSON file `users.json` in the project root. The
default file contains an `admin` account with the role `system_admin`.
`auth.py` reads and writes this file when users are created or updated. Each
record stores the username, tenant, role, assigned agents and a bcrypt-hashed
password.

## Tenants and Agents

Tenants logically group agents, vector stores and configuration files. The
`DEFAULT_TENANT` and `DEFAULT_AGENT` constants in `config.py` define the fallback
names (`public` and `default`). Admin users may manage multiple tenants and
agents via the API or CLI.

