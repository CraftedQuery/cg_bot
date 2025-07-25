"""
vectorstore.py - Vector store operations and RAG functionality
"""
from typing import Dict, List, Tuple
from pathlib import Path
import os

from fastapi import HTTPException
import json

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _IMPORT_ERROR = None
except ModuleNotFoundError as e:  # pragma: no cover - optional deps missing
    OpenAIEmbeddings = None  # type: ignore
    FAISS = None  # type: ignore
    RecursiveCharacterTextSplitter = None  # type: ignore
    _IMPORT_ERROR = e

from .config import store_path
from .embedding import get_embedding_model
from .database import log_llm_event

# Text splitter for document chunking
if RecursiveCharacterTextSplitter:
    TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
else:  # pragma: no cover - missing optional deps
    TEXT_SPLITTER = None

# Vector store cache
_vec_cache: Dict[str, "FAISS"] = {}


def _require_deps():
    """Ensure optional vector store dependencies are installed."""
    if _IMPORT_ERROR is not None:
        raise HTTPException(
            500,
            f"Missing optional dependency '{_IMPORT_ERROR.name}'. Install requirements to enable vector store.",
        )


def get_vector_store(tenant: str, agent: str) -> FAISS:
    """Get or load vector store for a tenant/agent"""
    _require_deps()
    cache_key = f"{tenant}/{agent}"
    path = store_path(tenant, agent)
    
    # Check cache first
    if cache_key in _vec_cache:
        return _vec_cache[cache_key]
    
    # Check if vector store exists
    if not path.exists():
        raise HTTPException(404, "Vector store missing; run ingest")
    
    # Determine embedding provider
    meta = {"provider": "openai", "model": None}
    meta_file = path / "meta.json"
    if meta_file.exists():
        try:
            meta.update(json.loads(meta_file.read_text()))
        except Exception:
            pass

    emb = get_embedding_model(meta["provider"], meta.get("model"))

    # Load and cache
    _vec_cache[cache_key] = FAISS.load_local(
        str(path),
        emb,
        allow_dangerous_deserialization=True,
    )
    return _vec_cache[cache_key]


def clear_cache(tenant: str, agent: str):
    """Clear vector store cache for a tenant/agent"""
    _require_deps()
    cache_key = f"{tenant}/{agent}"
    if cache_key in _vec_cache:
        del _vec_cache[cache_key]


def search_documents(
    tenant: str,
    agent: str,
    query: str,
    k: int = 4
) -> List[Tuple[str, Dict, float]]:
    """Search for relevant documents"""
    _require_deps()
    db = get_vector_store(tenant, agent)
    docs = db.similarity_search_with_score(query, k=k)
    
    # Format results
    results = []
    for doc, score in docs:
        results.append((
            doc.page_content,
            doc.metadata,
            score
        ))
    
    return results


def create_vector_store(
    tenant: str,
    agent: str,
    texts: List[str],
    metadatas: List[Dict],
    provider: str = "openai",
    model: str | None = None,
) -> bool:
    """Create a new vector store from texts and metadata"""
    _require_deps()
    try:
        path = store_path(tenant, agent)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings
        emb = get_embedding_model(provider, model)
        
        # Create and save vector store
        vec_store = FAISS.from_texts(texts, emb, metadatas=metadatas)
        vec_store.save_local(str(path))

        # Log embedding events for each unique file
        unique_sources = {
            m.get("source") for m in metadatas if m.get("source") is not None
        }
        for src in unique_sources:
            log_llm_event(
                f"{provider}-embed",
                "success",
                tenant=tenant,
                agent=agent,
                model=model,
                description=src,
            )
        (path / "meta.json").write_text(json.dumps({"provider": provider, "model": model}))
        
        # Clear cache to force reload
        clear_cache(tenant, agent)
        
        return True
    except Exception as e:
        raise HTTPException(500, f"Failed to create vector store: {str(e)}")


def update_vector_store(
    tenant: str,
    agent: str,
    texts: List[str],
    metadatas: List[Dict],
    provider: str = "openai",
    model: str | None = None,
) -> bool:
    """Append texts to an existing vector store or create a new one."""
    _require_deps()
    try:
        path = store_path(tenant, agent)
        path.mkdir(parents=True, exist_ok=True)

        meta_file = path / "meta.json"
        current_meta = {}
        if meta_file.exists():
            try:
                current_meta = json.loads(meta_file.read_text())
            except Exception:
                current_meta = {}

        meta_provider = current_meta.get("provider", provider)
        meta_model = current_meta.get("model", model)

        emb = get_embedding_model(meta_provider, meta_model)

        if path.exists() and any(path.iterdir()):
            vec_store = FAISS.load_local(
                str(path),
                emb,
                allow_dangerous_deserialization=True,
            )
            vec_store.add_texts(texts, metadatas=metadatas)
        else:
            vec_store = FAISS.from_texts(texts, emb, metadatas=metadatas)

        vec_store.save_local(str(path))

        unique_sources = {
            m.get("source") for m in metadatas if m.get("source") is not None
        }
        for src in unique_sources:
            log_llm_event(
                f"{meta_provider}-embed",
                "success",
                tenant=tenant,
                agent=agent,
                model=meta_model,
                description=src,
            )

        meta_file.write_text(json.dumps({"provider": meta_provider, "model": meta_model}))

        clear_cache(tenant, agent)

        return True
    except Exception as e:
        raise HTTPException(500, f"Failed to update vector store: {str(e)}")


def chunk_text(text: str) -> List[str]:
    """Chunk text into smaller pieces for vectorization"""
    _require_deps()
    return TEXT_SPLITTER.split_text(text)


def get_vector_store_size(tenant: str, agent: str) -> int:
    """Return the size in bytes of the vector store for a tenant/agent."""
    path = store_path(tenant, agent)
    if not path.exists():
        return 0

    if path.is_file():
        return os.path.getsize(path)

    size = 0
    for p in path.rglob("*"):
        if p.is_file():
            size += os.path.getsize(p)
    return size
