"""
routers/admin_routes.py - Admin interface endpoints
"""

from pathlib import Path
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["admin"])


@router.get("/admin", response_class=HTMLResponse)
async def get_admin_interface():
    """Redirect to admin interface"""
    return HTMLResponse(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Redirecting...</title>
        <script>
            window.location.href = '/admin.html';
        </script>
    </head>
    <body>
        <p>Redirecting to admin interface...</p>
    </body>
    </html>
    """
    )


@router.get("/admin.html", response_class=HTMLResponse)
async def serve_admin_html():
    """Serve the admin interface HTML file"""
    try:
        admin_html_path = Path("static/admin.html")
        if admin_html_path.exists():
            return HTMLResponse(admin_html_path.read_text())
        else:
            return HTMLResponse(
                """
            <!DOCTYPE html>
            <html>
            <head><title>Admin Interface Not Found</title></head>
            <body>
                <h1>Admin Interface</h1>
                <p>Please save the admin HTML interface as 'static/admin.html' in your project directory.</p>
                <p>You can access the API documentation at <a href="/docs">/docs</a></p>
            </body>
            </html>
            """
            )
    except Exception as e:
        return HTMLResponse(f"<h1>Error loading admin interface</h1><p>{str(e)}</p>")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    from datetime import datetime, timezone

    openai_status = "failed"
    anthropic_status = "failed"
    openai_error = ""
    anthropic_error = ""

    if not os.getenv("OPENAI_API_KEY"):
        openai_error = "API key is missing"
    else:
        try:
            from ..llm import _get_openai_response

            _get_openai_response([{"role": "user", "content": "ping"}])
            openai_status = "ready"
        except Exception as e:
            openai_error = str(e)

    if not os.getenv("ANTHROPIC_API_KEY"):
        anthropic_error = "API key is missing"
    else:
        try:
            from ..llm import _get_anthropic_response

            _get_anthropic_response([{"role": "user", "content": "ping"}])
            anthropic_status = "ready"
        except Exception as e:
            anthropic_error = str(e)

    return {
        "status": "healthy",
        "version": "7.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "openai": openai_status,
        "openai_error": openai_error,
        "anthropic": anthropic_status,
        "anthropic_error": anthropic_error,
    }


@router.post("/llm_test")
async def llm_test():
    """Manually test connectivity to the configured LLM providers"""
    from datetime import datetime, timezone
    from ..llm import _get_openai_response, _get_anthropic_response
    from ..database import log_llm_event

    openai_error = None
    anthropic_error = None
    openai_status = "skipped"
    anthropic_status = "skipped"

    if os.getenv("OPENAI_API_KEY"):
        try:
            _get_openai_response([{"role": "user", "content": "ping"}])
            openai_status = "ready"
            log_llm_event("openai", "success", None, description="connectivity test")
        except Exception as e:
            openai_status = "failed"
            openai_error = str(e)
            log_llm_event("openai", "error", openai_error, description="connectivity test")
    else:
        openai_error = "API key is missing"
        log_llm_event("openai", "skipped", openai_error, description="connectivity test")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            _get_anthropic_response([{"role": "user", "content": "ping"}])
            anthropic_status = "ready"
            log_llm_event("anthropic", "success", None, description="connectivity test")
        except Exception as e:
            anthropic_status = "failed"
            anthropic_error = str(e)
            log_llm_event("anthropic", "error", anthropic_error, description="connectivity test")
    else:
        anthropic_error = "API key is missing"
        log_llm_event("anthropic", "skipped", anthropic_error, description="connectivity test")

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message": "LLM test completed",
        "openai": openai_status,
        "openai_error": openai_error,
        "anthropic": anthropic_status,
        "anthropic_error": anthropic_error,
    }


@router.get("/llm_logs")
async def get_llm_logs(limit: int = 100):
    """Retrieve recent LLM logs"""
    from ..database import get_db

    with get_db() as con:
        cur = con.execute(
            "SELECT ts, provider, status, tenant, agent, model, description, error_message FROM llm_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()

        logs = []
        for r in rows:
            ts, provider, status, tenant, agent, model, desc, error = r
            answer = None
            question = None
            if desc and "q:" in desc:
                question = desc.split("q:", 1)[1]
            if question:
                cur2 = con.execute(
                    "SELECT answer FROM chat_logs WHERE tenant = ? AND agent = ? AND question = ? ORDER BY id DESC LIMIT 1",
                    (tenant or "", agent or "", question),
                )
                row2 = cur2.fetchone()
                if row2:
                    answer = row2[0]

            logs.append(
                {
                    "ts": ts,
                    "provider": provider,
                    "status": status,
                    "tenant": tenant,
                    "agent": agent,
                    "model": model,
                    "description": desc,
                    "answer": answer,
                    "error": error,
                }
            )

    return {"logs": logs}


@router.get("/llm_models")
async def get_llm_models(provider: str = "anthropic"):
    """Return available models for a given LLM provider."""
    provider = provider.lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(400, "Anthropic API key not configured")

        import requests

        try:
            rsp = requests.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "accept": "application/json",
                },
                timeout=10,
            )
            rsp.raise_for_status()
            data = rsp.json()
            # The API changed the response format from {"models": [...]} to
            # {"data": [...]}. Handle either case to remain backward compatible.
            models = data.get("models") or data.get("data") or []
            names = [m.get("name") or m.get("id") for m in models]
        except Exception as e:  # pragma: no cover - network errors
            raise HTTPException(502, f"Failed to fetch models: {e}")

        return {"provider": provider, "models": names}

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(400, "OpenAI API key not configured")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        try:
            rsp = client.models.list()
            names = [m.id for m in rsp.data if "gpt" in m.id]
        except Exception as e:  # pragma: no cover - network errors
            raise HTTPException(502, f"Failed to fetch models: {e}")

        return {"provider": provider, "models": names}

    if provider in {"vertexai", "google"}:
        # Vertex AI does not currently provide an easy model listing API
        # Return a curated list of common chat models.
        names = [
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.0-pro",
        ]
        return {"provider": provider, "models": names}

    raise HTTPException(400, "Unknown provider")
