# File Checker and Embedding Background Process

This document describes how the file checking and embedding workflow operates in this project, and how to keep it running continuously.

## 1. Overview

- **`ingestion.py`** orchestrates the ingestion pipeline.
- **`utils/file_processors.py`** extracts text from uploaded files.
- **`vectorstore.py`** creates and stores embeddings in a FAISS index.

The ingestion pipeline accepts files from Google Drive, local paths or a sitemap and builds embeddings for a specific tenant/agent pair.

## 2. Ingestion Flow

The `ingest` function collects text chunks from the specified sources and saves them to the vector store. Core logic:

```python
# ingestion.py

def ingest(
    tenant: str,
    agent: str,
    sitemap: Optional[str] = None,
    drive: Optional[str] = None,
    files: Optional[List[Path]] = None,
    console: Optional[Console] = None,
):
    """Ingest content from various sources into the vector store."""
    texts, metadatas = [], []

    if console:
        console.print(Panel.fit(
            f"Starting ingestion for tenant [bold]{tenant}[/bold], agent [bold]{agent}[/bold]"
        ))

    if drive:
        texts_drive, metas_drive = _ingest_from_drive(drive, console)
        texts.extend(texts_drive)
        metadatas.extend(metas_drive)

    if files:
        texts_files, metas_files = _ingest_from_files(files, console)
        texts.extend(texts_files)
        metadatas.extend(metas_files)

    if sitemap:
        texts_sitemap, metas_sitemap = _ingest_from_sitemap(sitemap, console)
        texts.extend(texts_sitemap)
        metadatas.extend(metas_sitemap)

    if not texts:
        msg = "Nothing to ingest"
        if console:
            console.print(f"[yellow]{msg}[/yellow]")
        else:
            print(msg)
        return

    if console:
        console.print(f"Processing [bold]{len(texts)}[/bold] text chunks into vector store...")

    create_vector_store(tenant, agent, texts, metadatas)

    msg = f"Vector store created/updated successfully"
    if console:
        console.print(f"[green]{msg}[/green]")
    else:
        print(msg)
```

The actual embedding happens inside `create_vector_store`:

```python
# vectorstore.py

def create_vector_store(
    tenant: str,
    agent: str,
    texts: List[str],
    metadatas: List[Dict]
) -> bool:
    """Create a new vector store from texts and metadata"""
    _require_deps()
    try:
        path = store_path(tenant, agent)
        path.mkdir(parents=True, exist_ok=True)

        emb = OpenAIEmbeddings()
        vec_store = FAISS.from_texts(texts, emb, metadatas=metadatas)
        vec_store.save_local(str(path))

        clear_cache(tenant, agent)
        return True
    except Exception as e:
        raise HTTPException(500, f"Failed to create vector store: {str(e)}")
```

Files are broken into chunks before embedding:

```python
# utils/file_processors.py

# Import here to avoid circular dependency
from ..vectorstore import chunk_text

# Chunk the text
chunks = chunk_text(raw_text)

# Create metadata for each chunk
metadatas = [{"source": filename} for _ in chunks]
```

## 3. Background File Checker

To automate ingestion you can run a small loop that checks for new jobs every minute:

```python
import time
from rag_chatbot.ingestion import ingest

while True:
    job = fetch_next_job()  # implement your own queue or folder watcher
    if job:
        ingest(job.tenant, job.agent, files=job.files)
    time.sleep(60)  # wait one minute
```

Replace `fetch_next_job()` with logic that discovers pending uploads (for example by reading a directory or polling a database).

## 4. Running in the Background

Save the checker as `file_checker.py` and start it as a daemon:

```bash
nohup python file_checker.py &
```

Alternatively create a `systemd` service or a cron job to ensure the script runs on boot. Make sure the necessary environment variables (such as API keys) are available to the process.

## 5. Monitoring

Redirect stdout and stderr to a log file when running in the background. Reviewing these logs helps diagnose any ingestion or embedding failures.
