"""
llm.py - LLM provider integrations
"""
import os
import time
from typing import List, Dict, Any

# Import the logging function without assuming the module is part of a package.
try:
    # When llm.py is imported as part of the package (e.g. rag_chatbot.llm)
    from .database import log_llm_event
except Exception:  # pragma: no cover - fallback for direct script execution
    # When executed as a stand-alone module (e.g. `python llm.py`)
    from database import log_llm_event

from openai import OpenAI


def get_llm_response(
    messages: List[Dict],
    provider: str = "openai",
    model: str = None,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """Get response from selected LLM provider"""
    start_time = time.time()
    tokens_in = _estimate_tokens(messages)
    
    try:
        if provider == "openai":
            response = _get_openai_response(messages, model, temperature)
        elif provider == "anthropic":
            response = _get_anthropic_response(messages, model, temperature)
        elif provider == "vertexai":
            response = _get_vertexai_response(messages, model, temperature)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
            
    except Exception as e:
        response = {
            "content": f"Error generating response: {str(e)}",
            "tokens_out": 0
        }
    
    latency = time.time() - start_time
    
    return {
        "content": response["content"],
        "latency": latency,
        "tokens_in": tokens_in,
        "tokens_out": response.get("tokens_out", _estimate_tokens([{"content": response["content"]}]))
    }


def _get_openai_response(messages: List[Dict], model: str = None, temperature: float = 0.3) -> Dict:
    """Get response from OpenAI"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        log_llm_event("openai", "error", "API key is missing")
        raise ValueError("API key is missing")

    client = OpenAI(api_key=api_key)
    model = model or "gpt-4o-mini"

    try:
        rsp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages
        )
        log_llm_event("openai", "success", None)
    except Exception as e:
        log_llm_event("openai", "error", str(e))
        raise

    return {
        "content": rsp.choices[0].message["content"],
        "tokens_out": rsp.usage.completion_tokens
    }


def _get_anthropic_response(messages: List[Dict], model: str = None, temperature: float = 0.3) -> Dict:
    """Get response from Anthropic"""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package not installed")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        log_llm_event("anthropic", "error", "API key is missing")
        raise ValueError("API key is missing")

    client = anthropic.Anthropic(api_key=api_key)
    model = model or "claude-3-opus-20240229"

    try:
        rsp = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=temperature,
            messages=[
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
        )
        log_llm_event("anthropic", "success", None)
    except Exception as e:
        log_llm_event("anthropic", "error", str(e))
        raise

    return {
        "content": rsp.content[0].text,
        "tokens_out": None  # Will be estimated
    }


def _get_vertexai_response(messages: List[Dict], model: str = None, temperature: float = 0.3) -> Dict:
    """Get response from Google Vertex AI"""
    try:
        from vertexai.generative_models import GenerativeModel
    except ImportError:
        raise ImportError("google-cloud-aiplatform package not installed")
    
    model_name = model or "gemini-1.5-pro"
    model = GenerativeModel(model_name)
    
    response = model.generate_content([
        {"role": m["role"], "parts": [{"text": m["content"]}]} 
        for m in messages
    ])
    
    return {
        "content": response.text,
        "tokens_out": None  # Will be estimated
    }


def _estimate_tokens(messages: List[Dict]) -> int:
    """Roughly estimate token count based on character count"""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4  # rough estimate: 4 chars per token

