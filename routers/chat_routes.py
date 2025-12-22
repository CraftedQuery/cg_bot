"""
routers/chat_routes.py - Chat and RAG endpoints
"""
import json
import logging
from typing import Any
from fastapi import APIRouter, Request, Query, Depends, HTTPException

from ..models import ChatRequest, ChatResponse, User
from ..auth import get_current_active_user
from ..config import DEFAULT_TENANT, DEFAULT_AGENT, load_config
from ..rag_pipeline import run_legal_rag
from ..llm import get_llm_response
from ..database import (
    log_answer_evaluation,
    log_chat,
    log_question_evaluation,
    update_feedback,
    update_question_evaluation_decision,
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
    
    def _normalize_status(raw_status: str | None) -> str:
        """Map evaluator responses into the canonical status set."""

        normalized = (raw_status or "").strip().lower()
        if normalized in {"reject", "rejected", "deny", "denied", "blocked"}:
            return "Rejected"
        if normalized in {"suggest", "suggestion", "revise", "rewrite", "rephrase"}:
            return "Suggest"
        return "Pass"

    def _should_proceed(normalized_status: str, proceed_flag: Any) -> bool:
        """Determine whether to continue the pipeline based on evaluator output."""

        if proceed_flag is None:
            return normalized_status == "Pass"
        if isinstance(proceed_flag, str):
            return proceed_flag.strip().lower() in {
                "true",
                "1",
                "yes",
                "y",
                "pass",
                "allow",
                "approved",
                "proceed",
            }
        return bool(proceed_flag)

    def _parse_question_evaluation(content: str):
        """Interpret the evaluator response as structured JSON."""

        default_status = "Pass"
        default = {
            "status": default_status,
            "proceed": True,
            "evaluation_summary": None,
            "reason": None,
            "suggested_question": None,
            "original_question": q,
            "criteria_met": None,
            "criteria_failed": None,
            "user_message": None,
            "raw_status": None,
        }

        try:
            parsed = json.loads(content)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if not isinstance(parsed, dict):
                return default
        except Exception:
            return default

        raw_status = parsed.get("status")
        status = _normalize_status(raw_status)
        proceed = _should_proceed(status, parsed.get("proceed"))

        if raw_status and status == "Pass" and raw_status.strip().lower() not in {"pass", "suggest", "reject", "rejected"}:
            logger.warning("Unexpected question evaluator status '%s'; defaulting to 'Pass'", raw_status)

        return {
            "status": status,
            "proceed": proceed,
            "evaluation_summary": parsed.get("evaluation_summary"),
            "reason": parsed.get("reason"),
            "suggested_question": parsed.get("suggested_question"),
            "original_question": parsed.get("original_question", q),
            "criteria_met": parsed.get("criteria_met"),
            "criteria_failed": parsed.get("criteria_failed"),
            "user_message": parsed.get("user_message"),
            "raw_status": raw_status,
        }

    # Optional question evaluation stage
    question_eval_id = req.question_evaluation_id
    question_eval_summary: dict | None = None
    if stage_configs["question_evaluator"].get("enabled") and not req.skip_question_evaluation:
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
                stage="question_evaluator",
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

        parsed_eval = _parse_question_evaluation(qe_result.get("content", ""))
        eval_status = parsed_eval.get("status", "Pass")
        qe_scores = qe_result.get("scores") if isinstance(qe_result, dict) else None
        qe_flags = qe_result.get("flags") if isinstance(qe_result, dict) else None
        question_eval_id = log_question_evaluation(
            tenant=tenant,
            agent=agent,
            session_id=session_id,
            conversation_id=None,
            original_question=q,
            evaluation_result=eval_status,
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
            evaluation_status=eval_status,
            reason=parsed_eval.get("evaluation_summary") or parsed_eval.get("reason"),
            suggested_question=parsed_eval.get("suggested_question"),
            proceeded=parsed_eval.get("proceed"),
            proceed_recommendation=parsed_eval.get("proceed"),
        )

        question_eval_summary = {
            **parsed_eval,
            "question_evaluation_id": question_eval_id,
        }

        if eval_status == "Rejected":
            return {
                "reply": parsed_eval.get("user_message")
                or parsed_eval.get("evaluation_summary")
                or parsed_eval.get("reason")
                or "The question was rejected by the evaluator.",
                "sources": [],
                "question_evaluation": question_eval_summary,
            }

        if eval_status == "Suggest" and not parsed_eval.get("proceed"):
            return {
                "reply": "We suggest refining your question for better results.",
                "sources": [],
                "question_evaluation": question_eval_summary,
            }

    elif req.skip_question_evaluation and question_eval_id:
        update_question_evaluation_decision(
            question_eval_id,
            user_choice=req.question_decision,
            proceeded=True,
            final_question=q,
        )
        decision_status = _normalize_status(req.question_decision or "pass")
        question_eval_summary = {
            "status": decision_status,
            "proceed": True,
            "question_evaluation_id": question_eval_id,
            "original_question": q,
            "suggested_question": None,
            "reason": None,
        }

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

    main_stage_cfg = stage_configs["main_rag"] or {}
    if not main_stage_cfg.get("enabled", True):
        rag = {"reply": "Main RAG bot is disabled for this agent.", "sources": [], "evidence": [], "answer_json": None}
        llm_result = {
            "content": rag["reply"],
            "latency": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "error": None,
            "provider": main_stage_cfg.get("provider") or cfg.get("llm_provider", "openai"),
            "model": main_stage_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini"),
        }
    else:
        retrieval_cfg = cfg.get("retrieval", {}) if isinstance(cfg.get("retrieval", {}), dict) else {}
        hyde_cfg = cfg.get("hyde", {}) if isinstance(cfg.get("hyde", {}), dict) else {}

        try:
            rag_result = run_legal_rag(
                tenant=tenant,
                agent=agent,
                question=q,
                user=current_user.username,
                language=language,
                hyde_enabled=bool(hyde_cfg.get("enabled", True)),
                hyde_provider=str(hyde_cfg.get("provider", "anthropic")),
                hyde_model=str(hyde_cfg.get("model", "claude-3-5-sonnet-20240620")),
                hyde_temperature=float(hyde_cfg.get("temperature", 0.2)),
                hyde_max_tokens=hyde_cfg.get("max_tokens", 400),
                hyde_system_prompt=hyde_cfg.get("system_prompt"),
                retrieval_mode=str(retrieval_cfg.get("mode", "mmr")),
                mmr_lambda_mult=float(retrieval_cfg.get("lambda_mult", 0.6)),
                mmr_fetch_k=int(retrieval_cfg.get("fetch_k", 50)),
                final_k=int(retrieval_cfg.get("k", 8)),
                answer_provider=main_stage_cfg.get("provider") or cfg.get("llm_provider", "openai"),
                answer_model=main_stage_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini"),
                answer_temperature=float(main_stage_cfg.get("temperature", cfg.get("temperature", 0.3))),
                answer_max_tokens=main_stage_cfg.get("max_tokens"),
                answer_system_prompt=main_stage_cfg.get("system_prompt"),
                structural_requirements=main_stage_cfg.get("structural_requirements"),
                json_repair_prompt=main_stage_cfg.get("json_repair_prompt"),
            )
        except HTTPException as exc:
            # Translate missing vector store into a user-friendly message.
            if exc.status_code == 404 and "Vector store missing" in str(exc.detail):
                raise HTTPException(
                    status_code=400,
                    detail="No documents are available for this matter yet. Please upload files first.",
                ) from exc
            raise
        except Exception as exc:
            logger.exception("RAG pipeline failed")
            raise HTTPException(
                status_code=500,
                detail="Sorry—something went wrong while generating your answer. Please try again.",
            ) from exc

        llm_result = {
            "content": rag_result.reply,
            "latency": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "error": None,
            "provider": main_stage_cfg.get("provider") or cfg.get("llm_provider", "openai"),
            "model": main_stage_cfg.get("model") or cfg.get("llm_model", "gpt-4o-mini"),
        }

        rag = {
            "reply": rag_result.reply,
            "sources": rag_result.sources,
            "evidence": rag_result.evidence,
            "answer_json": rag_result.answer_json,
        }

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
                stage="answer_evaluator",
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
        sources=json.dumps(rag.get("sources", [])),
        latency=llm_result.get("latency") or 0,
        tokens_in=llm_result.get("tokens_in") or 0,
        tokens_out=llm_result.get("tokens_out") or 0,
        user_ip=request.client.host,
        question_evaluation_id=question_eval_id,
        answer_evaluation_id=answer_eval_id,
    )

    if chat_id:
        from ..database import link_stage_conversation

        link_stage_conversation(chat_id, question_eval_id, answer_eval_id)

    return {
        "reply": rag.get("reply", llm_result["content"]),
        "sources": rag.get("sources", []),
        "question_evaluation": question_eval_summary,
        "evidence": rag.get("evidence", []),
        "answer_json": rag.get("answer_json"),
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
