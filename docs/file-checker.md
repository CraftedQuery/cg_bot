# File Checker and Embedding Background Process

This document explains how ingestion and embedding work in the chatbot and how you can run the workflow continuously in the background.

## 1. Core components

- **`ingestion.py`** – Entry point that coordinates sources (Google Drive, sitemap URLs, local files), tracks OCR usage, and calls the vector store layer.
- **`utils/file_processors.py`** – Extracts text from PDFs, DOCX, Markdown, plain text, and images (with OCR when needed) while returning per-chunk metadata.
- **`vectorstore.py`** – Chunks text, builds embeddings via `embedding.py`, and writes FAISS indexes alongside a `meta.json` that records the provider/model.

## 2. Ingestion flow

High-level logic from `ingestion.py`:

```python
ocr_info = {}
texts, metadatas = [], []

# Google Drive
texts_drive, metas_drive, drive_ocr = _ingest_from_drive(folder_id, console)
texts.extend(texts_drive); metadatas.extend(metas_drive); ocr_info.update(drive_ocr)

# Local files
texts_files, metas_files, file_ocr = _ingest_from_files(files, tenant, agent, embedding_provider, console)
texts.extend(texts_files); metadatas.extend(metas_files); ocr_info.update(file_ocr)

# Sitemaps
texts_sitemap, metas_sitemap = _ingest_from_sitemap(sitemap_url, console)
texts.extend(texts_sitemap); metadatas.extend(metas_sitemap)

update_vector_store(tenant, agent, texts, metadatas, provider=embedding_provider, model=embedding_model)
```

The function returns `ocr_info` so callers can mark which files required OCR when updating the `uploaded_files` table.

## 3. Running continuously

To process uploads in the background, wrap `ingest` in a simple loop or queue worker:

```python
import time
from pathlib import Path
from ingestion import ingest

while True:
    # Replace this with your queue or filesystem watcher
    jobs = discover_pending_jobs()  # returns [{tenant, agent, files, drive, sitemap}]
    for job in jobs:
        ingest(
            job["tenant"],
            job["agent"],
            files=[Path(p) for p in job.get("files", [])],
            drive=job.get("drive"),
            sitemap=job.get("sitemap"),
        )
    time.sleep(60)
```

Pair this worker with `add_uploaded_file` and `set_file_ocr_used` from `database.py` if you want to record progress and OCR usage in the SQLite tables. Running it under `nohup`, `systemd`, or a container entrypoint keeps ingestion alive after deployment.

## 4. Monitoring and recovery

- Use the CLI dashboard (`python cli.py dashboard`) to check vector store presence and interaction counts.
- Review `llm_logs` for embedding provider errors and `error_logs` for ingestion failures.
- Clearing a tenant/agent involves deleting its vector store directory and calling `delete_agent_data` in `database.py`.

These practices help keep embeddings current while providing auditability across tenants.
