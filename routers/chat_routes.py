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
from ..database import log_chat, update_feedback, is_template_file
from langdetect import detect, DetectorFactory

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

    # Separate template chunks from case content
    template_chunks = []
    doc_chunks = []
    for content, metadata, score in search_results:
        src = metadata.get("source")
        if src and is_template_file(tenant, agent, src):
            template_chunks.append(content)
        else:
            doc_chunks.append((content, metadata, score))

    # Build context from non-template search results
    ctx = "\n".join(content for content, _, _ in doc_chunks)
    
    # Detect the language of the user's question
    try:
        DetectorFactory.seed = 0
        lang_code = detect(q) if q else "en"
    except Exception:
        lang_code = "en"
    lang_map = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "zh-cn": "Chinese (Simplified)",
        "zh-tw": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "bn": "Bengali",
        "id": "Indonesian",
        "sw": "Swahili",
    }
    language = lang_map.get(lang_code.lower(), "English")

    # Create system message with context
    sys_content = cfg["system_prompt"]
    # Ensure the assistant responds in the language used by the user
    sys_content += f"\nPlease respond in {language}."
    if template_chunks:
        sys_content += "\n" + "\n".join(template_chunks)
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
    for _, metadata, _ in doc_chunks:
        key = (metadata.get("source"), metadata.get("page"), metadata.get("line"))
        if key[0] and key not in seen:
            citation = {"source": key[0]}
            if key[1] is not None:
                citation["page"] = key[1]
            if key[2] is not None:
                citation["line"] = key[2]
            if metadata.get("heading"):
                citation["heading"] = metadata["heading"]
            sources.append(citation)
            seen.add(key)
    
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
            "SELECT ts, question, answer FROM chat_logs WHERE tenant = ? AND agent = ? ORDER BY id DESC LIMIT ?",
            (tenant, agent, limit),
        )
        rows = cur.fetchall()

    return [
        {"timestamp": ts, "question": q, "answer": a}
        for ts, q, a in reversed(rows)
    ]
