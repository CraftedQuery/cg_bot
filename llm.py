"""
llm.py - LLM provider integrations
"""
import logging
import os
import time
from typing import List, Dict, Any

import requests

# Import the logging function without assuming the module is part of a package.
try:
    # When llm.py is imported as part of the package (e.g. rag_chatbot.llm)
    from .database import log_llm_event
except Exception:  # pragma: no cover - fallback for direct script execution
    # When executed as a stand-alone module (e.g. `python llm.py`)
    from database import log_llm_event

from openai import OpenAI

logger = logging.getLogger(__name__)


def get_llm_response(
    messages: List[Dict],
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.3,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    max_tokens: int | None = None,
    tenant: str | None = None,
    agent: str | None = None,
    user: str | None = None,
    question: str | None = None,
    description: str | None = None,
) -> Dict[str, Any]:
    """Get response from selected LLM provider"""
    start_time = time.time()
    tokens_in = _estimate_tokens(messages)
    logger.info("LLM request to provider=%s model=%s", provider, model)

    error_message = None
    try:
        if provider == "openai":
            response = _get_openai_response(messages, model, temperature, api_key=api_key, endpoint=endpoint, max_tokens=max_tokens)
        elif provider == "anthropic":
            response = _get_anthropic_response(messages, model, temperature, api_key=api_key, endpoint=endpoint, max_tokens=max_tokens)
        elif provider == "vertexai":
            response = _get_vertexai_response(messages, model, temperature, max_tokens=max_tokens)
        elif provider == "custom":
            response = _get_custom_response(messages, model, temperature, endpoint=endpoint, api_key=api_key, max_tokens=max_tokens)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        desc_parts = []
        if description:
            desc_parts.append(description)
        if user:
            desc_parts.append(f"user:{user}")
        if question:
            desc_parts.append(f"q:{question}")
        log_llm_event(
            provider,
            "success",
            None,
            tenant=tenant,
            agent=agent,
            model=model,
            description=" ".join(desc_parts) if desc_parts else None,
        )

    except Exception as e:
        error_message = str(e)
        desc_parts = []
        if description:
            desc_parts.append(description)
        if user:
            desc_parts.append(f"user:{user}")
        if question:
            desc_parts.append(f"q:{question}")
        log_llm_event(
            provider,
            "error",
            error_message,
            tenant=tenant,
            agent=agent,
            model=model,
            description=" ".join(desc_parts) if desc_parts else None,
        )
        logger.exception("LLM request failed")
        response = {
            "content": f"Error generating response: {error_message}",
            "tokens_out": 0,
        }

    latency = time.time() - start_time

    return {
        "content": response["content"],
        "latency": latency,
        "tokens_in": tokens_in,
        "tokens_out": response.get("tokens_out", _estimate_tokens([{"content": response["content"]}])),
        "error": error_message,
        "provider": provider,
        "model": model,
    }


def _get_openai_response(
    messages: List[Dict],
    model: str | None = None,
    temperature: float = 0.3,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    max_tokens: int | None = None,
) -> Dict:
    """Get response from OpenAI"""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("API key is missing")

    client_kwargs = {"api_key": api_key}
    if endpoint:
        client_kwargs["base_url"] = endpoint
    client = OpenAI(**client_kwargs)
    model = model or "gpt-4o-mini"

    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise

    return {
        "content": rsp.choices[0].message.content,
        "tokens_out": rsp.usage.completion_tokens,
    }


def _get_anthropic_response(
    messages: List[Dict],
    model: str | None = None,
    temperature: float = 0.3,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    max_tokens: int | None = None,
) -> Dict:
    """Get response from Anthropic"""
    try:
        import anthropic
        NOT_GIVEN = getattr(anthropic, "NOT_GIVEN", None)
    except ImportError:
        raise ImportError("anthropic package not installed")

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("API key is missing")

    client_kwargs = {"api_key": api_key}
    if endpoint:
        client_kwargs["base_url"] = endpoint
    client = anthropic.Anthropic(**client_kwargs)
    model = model or "claude-3-opus-20240229"

    try:
        system_parts = []
        filtered = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                filtered.append({"role": m.get("role"), "content": m.get("content", "")})
        system_prompt = "\n".join(system_parts) if system_parts else None

        rsp = client.messages.create(
            model=model,
            max_tokens=max_tokens or 1000,
            temperature=temperature,
            system=system_prompt if system_prompt is not None else NOT_GIVEN,
            messages=filtered,
        )
    except Exception as e:
        raise

    return {
        "content": rsp.content[0].text,
        "tokens_out": None  # Will be estimated
    }


def _get_vertexai_response(
    messages: List[Dict],
    model: str | None = None,
    temperature: float = 0.3,
    *,
    max_tokens: int | None = None,
) -> Dict:
    """Get response from Google Vertex AI"""
    try:
        from vertexai.generative_models import GenerativeModel
    except ImportError:
        raise ImportError("google-cloud-aiplatform package not installed")
    
    model_name = model or "gemini-1.5-pro"
    model = GenerativeModel(model_name)
    
    generation_config = {"temperature": temperature}
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens

    response = model.generate_content(
        [
            {"role": m["role"], "parts": [{"text": m.get("content", "")}]}  # type: ignore[arg-type]
            for m in messages
        ],
        generation_config=generation_config,
    )

    return {
        "content": response.text,
        "tokens_out": None  # Will be estimated
    }


def _get_custom_response(
    messages: List[Dict],
    model: str | None,
    temperature: float = 0.3,
    *,
    endpoint: str | None,
    api_key: str | None,
    max_tokens: int | None,
) -> Dict:
    """Call a custom text generation endpoint."""

    if not endpoint:
        raise ValueError("Custom provider requires an endpoint URL")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    rsp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    rsp.raise_for_status()
    data = rsp.json()
    content = data.get("content") or data.get("reply") or data.get("message")
    if not content:
        content = str(data)

    tokens_out = data.get("tokens_out")
    usage = data.get("usage") if isinstance(data, dict) else None
    if tokens_out is None and isinstance(usage, dict):
        tokens_out = usage.get("completion_tokens")

    return {
        "content": content,
        "tokens_out": tokens_out,
    }


def _estimate_tokens(messages: List[Dict]) -> int:
    """Roughly estimate token count based on character count"""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4  # rough estimate: 4 chars per token

