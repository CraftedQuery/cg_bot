"""Embedding provider utilities"""
from typing import Optional

try:
    from .database import log_llm_event
except Exception:  # pragma: no cover
    from database import log_llm_event


def get_embedding_model(provider: str = "openai", model: Optional[str] = None):
    """Return a LangChain embedding model for the given provider."""
    provider = provider.lower()
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            log_llm_event("openai-embed", "error", str(e), model=model or "text-embedding-3-small")
            raise ImportError("langchain_openai package not installed") from e
        log_llm_event("openai-embed", "success", None, model=model or "text-embedding-3-small")
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    elif provider == "anthropic":
        try:
            from langchain_community.embeddings import AnthropicEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            log_llm_event("anthropic-embed", "error", str(e), model=model or "claude-3-embedding-001")
            raise ImportError("Anthropic dependencies not installed") from e
        log_llm_event("anthropic-embed", "success", None, model=model or "claude-3-embedding-001")
        return AnthropicEmbeddings(model=model or "claude-3-embedding-001")
    elif provider in {"vertexai", "google"}:
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            log_llm_event("vertexai-embed", "error", str(e), model=model or "textembedding-gecko@001")
            raise ImportError("google-cloud-aiplatform package not installed") from e
        log_llm_event("vertexai-embed", "success", None, model=model or "textembedding-gecko@001")
        return VertexAIEmbeddings(model_name=model or "textembedding-gecko@001")
    elif provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            log_llm_event("local-embed", "error", str(e), model=model or "sentence-transformers/all-MiniLM-L6-v2")
            raise ImportError("sentence-transformers package not installed") from e
        log_llm_event("local-embed", "success", None, model=model or "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    else:
        log_llm_event("embedding", "error", f"Unknown provider: {provider}")
        raise ValueError(f"Unknown embedding provider: {provider}")
