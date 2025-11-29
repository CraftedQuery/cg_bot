"""
routers/chat_routes.py - Chat and RAG endpoints
"""
import json
import logging
from fastapi import APIRouter, Request, Query, Depends, HTTPException

from ..models import ChatRequest, ChatResponse, User
from ..auth import get_current_active_user
from ..config import DEFAULT_TENANT, DEFAULT_AGENT, load_config
from ..vectorstore import search_documents
from ..llm import get_llm_response
from ..database import (
    log_answer_evaluation,
    log_chat,
    log_question_evaluation,
    update_feedback,
    is_template_file,
)
from langdetect import detect, DetectorFactory

logger = logging.getLogger(__name__)

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

    session_id = request.headers.get("X-Session-Id", "anon")

    stage_configs = {
        "question_evaluator": cfg.get("question_evaluator", {}),
        "main_rag": cfg.get("main_rag", {}),
        "answer_evaluator": cfg.get("answer_evaluator", {}),
    }
    
    # Get the latest user question
    q = next((m["content"] for m in reversed(req.messages) if m["role"] == "user"), "")
    
    # Optional question evaluation stage
    question_eval_id = None
    if stage_configs["question_evaluator"].get("enabled"):
        try:
            qe_cfg = stage_configs["question_evaluator"]
            qe_messages = []
            if qe_cfg.get("system_prompt"):
                qe_messages.append({"role": "system", "content": qe_cfg.get("system_prompt", "")})
            qe_messages.append({"role": "user", "content": q})

            qe_result = get_llm_response(
                messages=qe_messages,
                provider=qe_cfg.get("provider", "openai"),
                model=qe_cfg.get("model"),
                temperature=qe_cfg.get("temperature", 0.3),
                api_key=qe_cfg.get("api_key") or None,
                endpoint=qe_cfg.get("endpoint") or None,
                max_tokens=qe_cfg.get("max_tokens"),
                tenant=tenant,
                agent=agent,
                user=current_user.username,
                question=q,
                description="question_evaluator",
            )
        except Exception as exc:  # Defensive: continue main flow
            logger.exception("Question evaluation failed")
            qe_result = {
                "content": f"Question evaluation failed: {exc}",
                "tokens_out": None,
                "latency": None,
                "error": str(exc),
                "provider": stage_configs["question_evaluator"].get("provider"),
                "model": stage_configs["question_evaluator"].get("model"),
            }

        qe_scores = qe_result.get("scores") if isinstance(qe_result, dict) else None
        qe_flags = qe_result.get("flags") if isinstance(qe_result, dict) else None
        question_eval_id = log_question_evaluation(
            tenant=tenant,
            agent=agent,
            session_id=session_id,
            conversation_id=None,
            original_question=q,
            evaluation_result=qe_result.get("content", ""),
            provider=qe_result.get("provider"),
            model=qe_result.get("model"),
            tokens_used=qe_result.get("tokens_out"),
            latency_ms=(qe_result.get("latency") * 1000) if qe_result.get("latency") is not None else None,
            error=qe_result.get("error"),
            username=current_user.username,
            evaluation_details=json.dumps(qe_result, default=str),
            flags=json.dumps(qe_flags, default=str) if qe_flags is not None else None,
            prompt=json.dumps(qe_messages, default=str),
            full_response=qe_result.get("content"),
            criteria_scores=json.dumps(qe_scores, default=str) if qe_scores is not None else None,
        )

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

    sources = []
    main_stage_cfg = stage_configs["main_rag"] or {}
    if not main_stage_cfg.get("enabled", True):
        llm_result = {
            "content": "Main RAG bot is disabled for this agent.",
            "latency": 0,
            "tokens_in": 0,
            "tokens_out": 0,
        }
    else:
        sys_content = main_stage_cfg.get("system_prompt") or cfg["system_prompt"]
        # Ensure the assistant responds in the language used by the user
        sys_content += f"\nPlease respond in {language}."
        if template_chunks and "template" in q.lower():
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
            provider=main_stage_cfg.get("provider") or cfg.get("llm_provider", "openai"),
            model=main_stage_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini"),
            temperature=main_stage_cfg.get("temperature", cfg.get("temperature", 0.3)),
            api_key=main_stage_cfg.get("api_key") or None,
            endpoint=main_stage_cfg.get("endpoint") or None,
            max_tokens=main_stage_cfg.get("max_tokens"),
            tenant=tenant,
            agent=agent,
            user=current_user.username,
            question=q,
            description="main_rag",
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

    # Optional answer evaluation stage
    answer_eval_id = None
    if stage_configs["answer_evaluator"].get("enabled"):
        try:
            ae_cfg = stage_configs["answer_evaluator"]
            ae_messages = []
            if ae_cfg.get("system_prompt"):
                ae_messages.append({"role": "system", "content": ae_cfg.get("system_prompt", "")})
            ae_messages.append({"role": "user", "content": llm_result.get("content", "")})

            ae_result = get_llm_response(
                messages=ae_messages,
                provider=ae_cfg.get("provider", "openai"),
                model=ae_cfg.get("model"),
                temperature=ae_cfg.get("temperature", 0.3),
                api_key=ae_cfg.get("api_key") or None,
                endpoint=ae_cfg.get("endpoint") or None,
                max_tokens=ae_cfg.get("max_tokens"),
                tenant=tenant,
                agent=agent,
                user=current_user.username,
                question=q,
                description="answer_evaluator",
            )
        except Exception as exc:
            logger.exception("Answer evaluation failed")
            ae_result = {
                "content": f"Answer evaluation failed: {exc}",
                "tokens_out": None,
                "latency": None,
                "error": str(exc),
                "provider": stage_configs["answer_evaluator"].get("provider"),
                "model": stage_configs["answer_evaluator"].get("model"),
            }

        answer_eval_id = log_answer_evaluation(
            tenant=tenant,
            agent=agent,
            session_id=session_id,
            conversation_id=None,
            original_answer=llm_result.get("content", ""),
            evaluation_result=ae_result.get("content", ""),
            provider=ae_result.get("provider"),
            model=ae_result.get("model"),
            tokens_used=ae_result.get("tokens_out"),
            latency_ms=(ae_result.get("latency") * 1000) if ae_result.get("latency") is not None else None,
            error=ae_result.get("error"),
            username=current_user.username,
            evaluation_details=json.dumps(ae_result, default=str),
            flags=json.dumps(ae_result.get("flags"), default=str) if isinstance(ae_result, dict) and ae_result.get("flags") is not None else None,
            issues=json.dumps(ae_result.get("issues"), default=str) if isinstance(ae_result, dict) and ae_result.get("issues") is not None else None,
            recommendations=json.dumps(ae_result.get("recommendations"), default=str) if isinstance(ae_result, dict) and ae_result.get("recommendations") is not None else None,
            selected_answer_provider=ae_result.get("selected_provider"),
            prompt=json.dumps(ae_messages, default=str),
            full_response=ae_result.get("content"),
            criteria_scores=json.dumps(ae_result.get("scores"), default=str) if isinstance(ae_result, dict) and ae_result.get("scores") is not None else None,
        )

    # Log the interaction
    chat_id = log_chat(
        tenant=tenant,
        agent=agent,
        session_id=session_id,
        question=q,
        answer=llm_result["content"],
        sources=json.dumps(sources),
        latency=llm_result["latency"],
        tokens_in=llm_result["tokens_in"],
        tokens_out=llm_result["tokens_out"],
        user_ip=request.client.host,
        question_evaluation_id=question_eval_id,
        answer_evaluation_id=answer_eval_id,
    )

    if chat_id:
        from ..database import link_stage_conversation

        link_stage_conversation(chat_id, question_eval_id, answer_eval_id)
    
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
