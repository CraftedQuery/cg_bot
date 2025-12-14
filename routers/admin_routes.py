"""
routers/admin_routes.py - Admin interface endpoints
"""

import json
import re
from datetime import datetime
from pathlib import Path
import os
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse

from ..auth import get_admin_user
from ..database import get_db, log_llm_event

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
    """Retrieve recent LLM logs with enhanced error details"""
    with get_db() as con:
        cur = con.execute(
            """SELECT ts, provider, status, tenant, agent, model, description, error_message,
                      user, question, stage, request_payload, response_payload, error_type,
                      error_details, latency_ms, tokens_in, tokens_out
               FROM llm_logs ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()

        logs = []
        for r in rows:
            (ts, provider, status, tenant, agent, model, desc, error,
             user, question, stage, request_payload, response_payload, error_type,
             error_details, latency_ms, tokens_in, tokens_out) = r
            
            # For backward compatibility, try to extract question/user from description if not directly stored
            if not question and desc and "q:" in desc:
                question = desc.split("q:", 1)[1].strip()
            if not user and desc and "user:" in desc:
                user_match = re.search(r"user:([^\s]+)", desc)
                if user_match:
                    user = user_match.group(1)
            
            # Try to get answer from chat_logs if not in response_payload (backward compatibility)
            answer = None
            if question:
                try:
                    cur2 = con.execute(
                        "SELECT answer FROM chat_logs WHERE tenant = ? AND agent = ? AND question = ? ORDER BY id DESC LIMIT 1",
                        (tenant or "", agent or "", question),
                    )
                    row2 = cur2.fetchone()
                    if row2:
                        answer = row2[0]
                except Exception:
                    pass  # Ignore errors in backward compatibility lookup
            
            # Parse JSON fields if they exist
            request_payload_parsed = None
            response_payload_parsed = None
            error_details_parsed = None
            
            try:
                if request_payload:
                    request_payload_parsed = json.loads(request_payload)
            except Exception:
                pass
            
            try:
                if response_payload:
                    response_payload_parsed = json.loads(response_payload)
                    # Extract answer from response payload if available
                    if not answer and response_payload_parsed and "content" in response_payload_parsed:
                        answer = response_payload_parsed.get("content")
            except Exception:
                pass
            
            try:
                if error_details:
                    error_details_parsed = json.loads(error_details)
            except Exception:
                pass

            logs.append(
                {
                    "ts": ts,
                    "provider": provider,
                    "status": status,
                    "tenant": tenant,
                    "agent": agent,
                    "model": model,
                    "description": desc,
                    "user": user,
                    "question": question,
                    "stage": stage,
                    "answer": answer,
                    "error": error,
                    "error_type": error_type,
                    "error_details": error_details_parsed,
                    "request_payload": request_payload_parsed,
                    "response_payload": response_payload_parsed,
                    "latency_ms": latency_ms,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }
            )

    return {"logs": logs}


def _parse_date_filter(date_str: str | None):
    """Convert a date string to datetime for filtering."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


@router.get("/question_evaluation_logs", dependencies=[Depends(get_admin_user)])
async def get_question_evaluation_logs(
    limit: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
    result: str | None = None,
    user: str | None = None,
    provider: str | None = None,
    search: str | None = None,
):
    """Return question evaluation pipeline logs with optional filters."""

    start_dt = _parse_date_filter(start_date)
    end_dt = _parse_date_filter(end_date)

    conditions = []
    params: list[str | int | float] = []
    if start_dt:
        conditions.append("ts >= ?")
        params.append(start_dt.isoformat())
    if end_dt:
        conditions.append("ts <= ?")
        params.append(end_dt.isoformat())
    if result:
        conditions.append("evaluation_result LIKE ?")
        params.append(f"%{result}%")
    if user:
        conditions.append("username = ?")
        params.append(user)
    if provider:
        conditions.append("provider = ?")
        params.append(provider)
    if search:
        conditions.append("(original_question LIKE ? OR evaluation_result LIKE ? OR full_response LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as con:
        cur = con.execute(
            f"""
            SELECT id, ts, tenant, agent, session_id, conversation_id, original_question, evaluation_result,
                   provider, model, tokens_used, latency_ms, error, username, evaluation_details, flags,
                   prompt, full_response, criteria_scores, evaluation_status, reason, suggested_question,
                   user_choice, proceeded, final_question, proceed_recommendation
            FROM question_evaluation_logs
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = cur.fetchall()

    def parse_json(value):
        try:
            return json.loads(value) if value else None
        except Exception:
            return value

    logs = [
        {
            "id": r[0],
            "ts": r[1],
            "tenant": r[2],
            "agent": r[3],
            "session_id": r[4],
            "conversation_id": r[5],
            "original_question": r[6],
            "evaluation_result": r[7],
            "provider": r[8],
            "model": r[9],
            "tokens_used": r[10],
            "latency_ms": r[11],
            "error": r[12],
            "username": r[13],
            "evaluation_details": parse_json(r[14]),
            "flags": parse_json(r[15]),
            "prompt": parse_json(r[16]),
            "full_response": r[17],
            "criteria_scores": parse_json(r[18]),
            "evaluation_status": r[19],
            "reason": r[20],
            "suggested_question": r[21],
            "user_choice": r[22],
            "proceeded": r[23],
            "final_question": r[24],
            "proceed_recommendation": r[25],
        }
        for r in rows
    ]

    return {"logs": logs}


@router.get("/answer_evaluation_logs", dependencies=[Depends(get_admin_user)])
async def get_answer_evaluation_logs(
    limit: int = 200,
    start_date: str | None = None,
    end_date: str | None = None,
    result: str | None = None,
    user: str | None = None,
    provider: str | None = None,
    search: str | None = None,
):
    """Return answer evaluation pipeline logs with optional filters."""

    start_dt = _parse_date_filter(start_date)
    end_dt = _parse_date_filter(end_date)

    conditions = []
    params: list[str | int | float] = []
    if start_dt:
        conditions.append("ts >= ?")
        params.append(start_dt.isoformat())
    if end_dt:
        conditions.append("ts <= ?")
        params.append(end_dt.isoformat())
    if result:
        conditions.append("evaluation_result LIKE ?")
        params.append(f"%{result}%")
    if user:
        conditions.append("username = ?")
        params.append(user)
    if provider:
        conditions.append("provider = ?")
        params.append(provider)
    if search:
        conditions.append("(original_answer LIKE ? OR evaluation_result LIKE ? OR full_response LIKE ? OR issues LIKE ?)")
        params.extend([f"%{search}%", f"%{search}%", f"%{search}%", f"%{search}%"])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    with get_db() as con:
        cur = con.execute(
            f"""
            SELECT id, ts, tenant, agent, session_id, conversation_id, original_answer, evaluation_result,
                   provider, model, tokens_used, latency_ms, error, username, evaluation_details, flags,
                   issues, recommendations, selected_answer_provider, prompt, full_response, criteria_scores
            FROM answer_evaluation_logs
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = cur.fetchall()

    def parse_json(value):
        try:
            return json.loads(value) if value else None
        except Exception:
            return value

    logs = [
        {
            "id": r[0],
            "ts": r[1],
            "tenant": r[2],
            "agent": r[3],
            "session_id": r[4],
            "conversation_id": r[5],
            "original_answer": r[6],
            "evaluation_result": r[7],
            "provider": r[8],
            "model": r[9],
            "tokens_used": r[10],
            "latency_ms": r[11],
            "error": r[12],
            "username": r[13],
            "evaluation_details": parse_json(r[14]),
            "flags": parse_json(r[15]),
            "issues": parse_json(r[16]),
            "recommendations": parse_json(r[17]),
            "selected_answer_provider": r[18],
            "prompt": parse_json(r[19]),
            "full_response": r[20],
            "criteria_scores": parse_json(r[21]),
        }
        for r in rows
    ]

    return {"logs": logs}



@router.get("/error_logs", dependencies=[Depends(get_admin_user)])

async def get_error_logs(limit: int = 100):
    """Retrieve recent application error logs"""
    from ..database import get_error_logs as db_get_error_logs

    rows = db_get_error_logs(limit)
    logs = [
        {
            "ts": ts,
            "endpoint": endpoint,
            "tenant": tenant,
            "agent": agent,
            "message": message,
        }
        for ts, endpoint, tenant, agent, message in rows
    ]
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


@router.post("/admin/update-styles", dependencies=[Depends(get_admin_user)])
async def update_global_styles(request: Request):
    """
    Update the global CSS stylesheet.
    Requires admin authentication.
    """
    try:
        data = await request.json()
        css_content = data.get("css", "")
        
        if not css_content:
            raise HTTPException(status_code=400, detail="CSS content is required")
        
        # Validate CSS (basic check)
        if len(css_content) > 1_000_000:  # 1MB limit
            raise HTTPException(status_code=400, detail="CSS file too large (max 1MB)")
        
        # Write to global.css
        css_path = Path("static/css/global.css")
        css_path.parent.mkdir(parents=True, exist_ok=True)
        
        css_path.write_text(css_content, encoding="utf-8")
        
        return {
            "status": "success",
            "message": "Global styles updated successfully",
            "size": len(css_content)
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")
    except HTTPException:
        # Re-raise HTTPExceptions (validation errors) without modification
        raise
    except Exception as e:
        from ..database import log_error
        log_error("update_global_styles", str(e), tenant=None, agent=None)
        raise HTTPException(status_code=500, detail=f"Failed to update styles: {str(e)}")
