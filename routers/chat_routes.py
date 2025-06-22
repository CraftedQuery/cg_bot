"""
routers/chat_routes.py - Chat and RAG endpoints
"""
import json
from fastapi import APIRouter, Request, Query, Depends, HTTPException

from ..models import ChatRequest, ChatResponse, User
from ..auth import get_current_active_user
from ..config import DEFAULT_TENANT, DEFAULT_AGENT, load_config
from ..vectorstore import search_documents
from ..llm import get_llm_response
from ..database import log_chat, update_feedback

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    current_user: User = Depends(get_current_active_user)
):
    """Main chat endpoint with RAG functionality"""
    
    # Check if user has access to this tenant
    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(
            status_code=403,
            detail="You don't have access to this tenant"
        )
    
    # Get configuration
    cfg = load_config(tenant, agent)
    
    # Get the latest user question
    q = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    
    # Search for relevant documents
    search_results = search_documents(tenant, agent, q)
    
    # Build context from search results
    ctx = "\n".join(content for content, _, _ in search_results)
    
    # Create system message with context
    sys_content = cfg["system_prompt"]
    if cfg.get("local_only", True):
        sys_content += "\nUse only the provided Context to answer. Do not search the internet."
    sys_content += "\nContext:\n" + ctx
    system_msg = {
        "role": "system",
        "content": sys_content
    }
    
    # Get response from LLM
    llm_result = get_llm_response(
        messages=[system_msg, *req.messages],
        provider=cfg.get("llm_provider", "openai"),
        model=cfg.get("llm_model", "gpt-4o-mini"),
        temperature=cfg.get("temperature", 0.3),
        tenant=tenant,
        agent=agent,
        user=current_user.username,
        question=q,
    )
    
    # Extract sources
    sources, seen = [], set()
    for _, metadata, _ in search_results:
        s = metadata.get("source", "")
        if s and s not in seen:
            sources.append({"source": s})
            seen.add(s)
    
    # Log the interaction
    log_chat(
        tenant=tenant,
        agent=agent,
        session_id=request.headers.get("X-Session-Id", "anon"),
        question=q,
        answer=llm_result["content"],
        sources=json.dumps(sources),
        latency=llm_result["latency"],
        tokens_in=llm_result["tokens_in"],
        tokens_out=llm_result["tokens_out"],
        user_ip=request.client.host
    )
    
    return {
        "reply": llm_result["content"],
        "sources": sources
    }


@router.post("/feedback/{chat_id}")
async def submit_feedback(
    chat_id: int,
    feedback: int,
    current_user: User = Depends(get_current_active_user)
):
    """Submit feedback for a chat interaction"""
    
    # Validate feedback score (1-5)
    if feedback < 1 or feedback > 5:
        raise HTTPException(
            status_code=400,
            detail="Feedback must be between 1 and 5"
        )
    
    # Update the feedback
    if not update_feedback(chat_id, feedback):
        raise HTTPException(
            status_code=404,
            detail="Chat log not found"
        )

    return {"message": "Feedback submitted successfully"}


@router.get("/history")
async def chat_history(
    tenant: str = Query(DEFAULT_TENANT),
    agent: str = Query(DEFAULT_AGENT),
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
):
    """Return recent chat history for a tenant/agent"""

    if current_user.tenant != "*" and current_user.tenant != tenant:
        raise HTTPException(403, "You don't have access to this tenant")

    from ..database import get_db

    with get_db() as con:
        cur = con.execute(
            "SELECT question, answer FROM chat_logs WHERE tenant = ? AND agent = ? ORDER BY id DESC LIMIT ?",
            (tenant, agent, limit),
        )
        rows = cur.fetchall()

    return [
        {"question": q, "answer": a}
        for q, a in reversed(rows)
    ]
