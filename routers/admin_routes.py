"""
routers/admin_routes.py - Admin interface endpoints
"""

from pathlib import Path
import os
from fastapi import APIRouter
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

    openai_error = None
    anthropic_error = None
    openai_status = "skipped"
    anthropic_status = "skipped"

    if os.getenv("OPENAI_API_KEY"):
        try:
            _get_openai_response([{"role": "user", "content": "ping"}])
            openai_status = "ready"
        except Exception as e:
            openai_status = "failed"
            openai_error = str(e)

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            _get_anthropic_response([{"role": "user", "content": "ping"}])
            anthropic_status = "ready"
        except Exception as e:
            anthropic_status = "failed"
            anthropic_error = str(e)

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
            "SELECT ts, provider, status, error_message FROM llm_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()

    return {
        "logs": [
            {
                "ts": r[0],
                "provider": r[1],
                "status": r[2],
                "error": r[3],
            }
            for r in rows
        ]
    }
