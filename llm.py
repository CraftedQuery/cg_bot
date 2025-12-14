"""
llm.py - LLM provider integrations
"""
import logging
import os
import time
import json
import traceback
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


def _classify_error(exception: Exception) -> tuple[str, dict]:
    """Classify error type and extract detailed error information"""
    error_type = "unknown"
    error_details = {
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "stack_trace": traceback.format_exc(),
    }
    
    error_str = str(exception).lower()
    error_class = type(exception).__name__.lower()
    
    # Check for rate limiting
    if "rate limit" in error_str or "rate_limit" in error_str or "429" in error_str:
        error_type = "rate_limit"
        if hasattr(exception, "response"):
            try:
                error_details["http_status"] = getattr(exception.response, "status_code", None)
                if hasattr(exception.response, "json"):
                    error_details["api_error_response"] = exception.response.json()
            except Exception:
                pass
    
    # Check for timeout
    elif "timeout" in error_str or "timed out" in error_str:
        error_type = "timeout"
    
    # Check for authentication errors
    elif "auth" in error_str or "401" in error_str or "403" in error_str:
        error_type = "authentication_error"
        if hasattr(exception, "response"):
            try:
                error_details["http_status"] = getattr(exception.response, "status_code", None)
            except Exception:
                pass
    
    # Check for API errors (HTTP errors from providers)
    elif hasattr(exception, "response") or "http" in error_str or "api" in error_str:
        error_type = "api_error"
        if hasattr(exception, "response"):
            try:
                error_details["http_status"] = getattr(exception.response, "status_code", None)
                if hasattr(exception.response, "json"):
                    error_details["api_error_response"] = exception.response.json()
                elif hasattr(exception.response, "text"):
                    error_details["api_error_response"] = exception.response.text
            except Exception:
                pass
    
    # Check for validation/parsing errors
    elif "json" in error_str or "parse" in error_str or "validation" in error_str:
        error_type = "validation_error"
    
    # Check for provider-specific errors
    elif "model" in error_str and ("not found" in error_str or "invalid" in error_str):
        error_type = "provider_error"
    
    return error_type, error_details


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
    stage: str | None = None,
    optional: bool = False,
) -> Dict[str, Any]:
    """Get response from selected LLM provider with enhanced logging"""
    start_time = time.time()
    tokens_in = _estimate_tokens(messages)
    logger.info("LLM request to provider=%s model=%s stage=%s", provider, model, stage)

    # Prepare request payload for logging
    request_payload_dict = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_payload_dict["max_tokens"] = max_tokens
    if endpoint:
        request_payload_dict["endpoint"] = endpoint
    request_payload_json = json.dumps(request_payload_dict, indent=2)

    error_message = None
    error_type = None
    error_details = None
    response_payload_json = None
    tokens_out = 0

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

        # Prepare response payload for logging
        tokens_out = response.get("tokens_out", 0)
        response_payload_dict = {
            "content": response.get("content", ""),
            "tokens_out": tokens_out,
        }
        response_payload_json = json.dumps(response_payload_dict, indent=2)

        desc_parts = []
        if description:
            desc_parts.append(description)
        if user:
            desc_parts.append(f"user:{user}")
        if question:
            desc_parts.append(f"q:{question}")
        
        latency_ms = (time.time() - start_time) * 1000
        
        log_llm_event(
            provider,
            "success",
            None,
            tenant=tenant,
            agent=agent,
            model=model,
            description=" ".join(desc_parts) if desc_parts else None,
            user=user,
            question=question,
            stage=stage,
            request_payload=request_payload_json,
            response_payload=response_payload_json,
            error_type=None,
            error_details=None,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    except Exception as e:
        error_message = str(e)
        error_type, error_details_dict = _classify_error(e)
        error_details_json = json.dumps(error_details_dict, indent=2)
        
        desc_parts = []
        if description:
            desc_parts.append(description)
        if user:
            desc_parts.append(f"user:{user}")
        if question:
            desc_parts.append(f"q:{question}")
        
        # Optional stages (e.g., HyDE) should not be treated as a hard error in logs.
        status = "skipped" if optional else "error"
        
        latency_ms = (time.time() - start_time) * 1000
        
        log_llm_event(
            provider,
            status,
            error_message,
            tenant=tenant,
            agent=agent,
            model=model,
            description=" ".join(desc_parts) if desc_parts else None,
            user=user,
            question=question,
            stage=stage,
            request_payload=request_payload_json,
            response_payload=response_payload_json,
            error_type=error_type,
            error_details=error_details_json,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=0,
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

    # Newer OpenAI models (gpt-5.x, o-series) require max_completion_tokens instead of max_tokens
    # Check if model name indicates a newer model that requires max_completion_tokens
    requires_max_completion_tokens = (
        model.startswith("gpt-5") or 
        model.startswith("o1") or
        model.startswith("o3")
    )
    
    def _from_chat_completions() -> Dict:
        create_kwargs = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if max_tokens is not None:
            if requires_max_completion_tokens:
                create_kwargs["max_completion_tokens"] = max_tokens
            else:
                create_kwargs["max_tokens"] = max_tokens

        rsp = client.chat.completions.create(**create_kwargs)
        return {
            "content": rsp.choices[0].message.content,
            "tokens_out": getattr(rsp.usage, "completion_tokens", None),
        }

    def _from_responses_api() -> Dict:
        # Some newer OpenAI models are served via the Responses API.
        if not hasattr(client, "responses"):
            raise RuntimeError("OpenAI client does not support Responses API")
        req: Dict[str, Any] = {
            "model": model,
            "input": [{"role": m.get("role"), "content": m.get("content", "")} for m in messages],
        }
        # Keep temperature consistent if supported by the server.
        req["temperature"] = temperature
        if max_tokens is not None:
            # Responses API uses `max_output_tokens`.
            req["max_output_tokens"] = max_tokens
        rsp = client.responses.create(**req)
        # `output_text` is the most stable accessor across SDK versions.
        content = getattr(rsp, "output_text", None)
        if content is None:
            # Defensive fallback for older SDK shapes.
            content = getattr(rsp, "text", "") or ""
        usage = getattr(rsp, "usage", None)
        tokens_out = None
        if usage is not None:
            tokens_out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        return {"content": content, "tokens_out": tokens_out}

    try:
        return _from_chat_completions()
    except Exception as e:
        # If we get a BadRequestError about max_tokens, retry with max_completion_tokens
        if "max_tokens" in str(e) and "max_completion_tokens" in str(e) and not requires_max_completion_tokens:
            try:
                # Force the retry path with `max_completion_tokens`.
                retry_kwargs: Dict[str, Any] = {
                    "model": model,
                    "temperature": temperature,
                    "messages": messages,
                }
                if max_tokens is not None:
                    retry_kwargs["max_completion_tokens"] = max_tokens
                rsp = client.chat.completions.create(**retry_kwargs)
                return {
                    "content": rsp.choices[0].message.content,
                    "tokens_out": getattr(rsp.usage, "completion_tokens", None),
                }
            except Exception as retry_error:
                raise retry_error from e

        # Fallback to Responses API for models that don't support chat.completions.
        msg = str(e).lower()
        if "responses" in msg or "does not support" in msg or "chat.completions" in msg:
            try:
                return _from_responses_api()
            except Exception as responses_error:
                # Preserve the original exception if fallback also fails.
                raise e from responses_error
        raise


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

