"""
vectorstore.py - Vector store operations and RAG functionality
"""
from typing import Dict, List, Tuple
from pathlib import Path

from fastapi import HTTPException
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import store_path

# Text splitter for document chunking
TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)

# Vector store cache
_vec_cache: Dict[str, FAISS] = {}


def get_vector_store(tenant: str, agent: str) -> FAISS:
    """Get or load vector store for a tenant/agent"""
    cache_key = f"{tenant}/{agent}"
    path = store_path(tenant, agent)
    
    # Check cache first
    if cache_key in _vec_cache:
        return _vec_cache[cache_key]
    
    # Check if vector store exists
    if not path.exists():
        raise HTTPException(404, "Vector store missing; run ingest")
    
    # Load and cache
    _vec_cache[cache_key] = FAISS.load_local(str(path), OpenAIEmbeddings())
    return _vec_cache[cache_key]


def clear_cache(tenant: str, agent: str):
    """Clear vector store cache for a tenant/agent"""
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
    metadatas: List[Dict]
) -> bool:
    """Create a new vector store from texts and metadata"""
    try:
        path = store_path(tenant, agent)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create embeddings
        emb = OpenAIEmbeddings()
        
        # Create and save vector store
        vec_store = FAISS.from_texts(texts, emb, metadatas=metadatas)
        vec_store.save_local(str(path))
        
        # Clear cache to force reload
        clear_cache(tenant, agent)
        
        return True
    except Exception as e:
        raise HTTPException(500, f"Failed to create vector store: {str(e)}")


def chunk_text(text: str) -> List[str]:
    """Chunk text into smaller pieces for vectorization"""
    return TEXT_SPLITTER.split_text(text)