"""Embedding provider utilities"""
from typing import Optional


def get_embedding_model(provider: str = "openai", model: Optional[str] = None):
    """Return a LangChain embedding model for the given provider."""
    provider = provider.lower()
    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            raise ImportError("langchain_openai package not installed") from e
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")
    elif provider == "anthropic":
        try:
            from langchain_community.embeddings import AnthropicEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            raise ImportError("Anthropic dependencies not installed") from e
        return AnthropicEmbeddings(model=model or "claude-3-embedding-001")
    elif provider in {"vertexai", "google"}:
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            raise ImportError("google-cloud-aiplatform package not installed") from e
        return VertexAIEmbeddings(model_name=model or "textembedding-gecko@001")
    elif provider == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ModuleNotFoundError as e:  # pragma: no cover - optional deps
            raise ImportError("sentence-transformers package not installed") from e
        return HuggingFaceEmbeddings(model_name=model or "sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
