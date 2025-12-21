"""
rag_pipeline.py

Production-oriented RAG pipeline helpers:
- HyDE query generation (Claude 3.5 Sonnet by default)
- MMR retrieval (FAISS via LangChain)
- Evidence packing with stable citation IDs
- Structured, citation-anchored answer generation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, ValidationError

from .llm import get_llm_response
try:
    # `retrieve_documents_mmr` may be absent in some test stubs.
    from .vectorstore import retrieve_documents_mmr, search_documents
except Exception:  # pragma: no cover - fallback for test stubs / partial installs
    from .vectorstore import search_documents  # type: ignore

    retrieve_documents_mmr = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CitationRef(BaseModel):
    """A reference to a specific evidence item by its stable ID."""

    citation_id: str = Field(..., description="Evidence ID like 'C1'")


class AnswerBullet(BaseModel):
    text: str = Field(..., min_length=1)
    citation_id: str = Field(..., pattern=r"^C\d+$")


class AnswerQuote(BaseModel):
    quote: str = Field(..., min_length=1)
    citation_id: str = Field(..., pattern=r"^C\d+$")


class StructuredAnswer(BaseModel):
    """Model-facing answer format (strict JSON)."""

    summary_bullets: list[AnswerBullet] = Field(default_factory=list)
    key_quotes: list[AnswerQuote] = Field(default_factory=list)
    limitations: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    citation_id: str
    source: str
    page: int | None
    line_start: int | None
    line_end: int | None
    heading: str | None
    quote: str


class RAGResult(BaseModel):
    """API-facing payload (UI sidebar can render `evidence`)."""

    reply: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    answer_json: dict[str, Any] | None = None


def _line_end_fallback(meta: dict[str, Any]) -> int | None:
    """Fallback when older vector stores don't have `line_end`."""

    line = meta.get("line")
    if isinstance(line, int):
        return line
    return None


def build_evidence_pack(
    chunks: Iterable[tuple[str, dict[str, Any]]],
    *,
    max_items: int,
) -> list[EvidenceItem]:
    """Assign stable citation IDs and normalize metadata."""

    items: list[EvidenceItem] = []
    for idx, (content, meta) in enumerate(chunks, start=1):
        if idx > max_items:
            break
        src = str(meta.get("source") or "").strip()
        if not src:
            src = "unknown"
        page = meta.get("page")
        page_i = page if isinstance(page, int) else None
        line_start = meta.get("line")
        line_start_i = line_start if isinstance(line_start, int) else None
        line_end = meta.get("line_end")
        line_end_i = line_end if isinstance(line_end, int) else _line_end_fallback(meta)
        heading = meta.get("heading")
        heading_s = heading if isinstance(heading, str) and heading.strip() else None
        quote = (content or "").strip()
        if not quote:
            continue

        items.append(
            EvidenceItem(
                citation_id=f"C{idx}",
                source=src,
                page=page_i,
                line_start=line_start_i,
                line_end=line_end_i,
                heading=heading_s,
                quote=quote,
            )
        )
    return items


def _format_pl(item: EvidenceItem) -> str:
    page = item.page
    ls = item.line_start
    le = item.line_end
    if page is None and ls is None and le is None:
        return ""
    if page is None:
        if ls is None:
            return ""
        if le is None or le == ls:
            return f"Line {ls}"
        return f"Line {ls}-{le}"
    if ls is None:
        return f"Page {page}"
    if le is None or le == ls:
        return f"Page {page}, Line {ls}"
    return f"Page {page}, Line {ls}-{le}"


def _render_citation_bracket(item: EvidenceItem) -> str:
    pl = _format_pl(item)
    if not pl:
        return f"[Source: {item.source}]"
    return f"[Source: {pl}]"


def render_answer_with_citations(
    answer: StructuredAnswer,
    evidence_by_id: dict[str, EvidenceItem],
) -> str:
    """Render final user-visible answer with forced bracket citations."""

    lines: list[str] = []

    if answer.summary_bullets:
        lines.append("Summary")
        for b in answer.summary_bullets:
            ev = evidence_by_id.get(b.citation_id)
            if not ev:
                # Defensive: keep output safe and non-hallucinated.
                continue
            # Add a machine-readable anchor for the UI to hook into.
            bracket = _render_citation_bracket(ev)
            lines.append(f"- {b.text.strip()} {bracket} {{cite:{ev.citation_id}}}")

    if answer.key_quotes:
        if lines:
            lines.append("")
        lines.append("Key quotes")
        for q in answer.key_quotes:
            ev = evidence_by_id.get(q.citation_id)
            if not ev:
                continue
            bracket = _render_citation_bracket(ev)
            quote_text = q.quote.strip()
            lines.append(f"- “{quote_text}” {bracket} {{cite:{ev.citation_id}}}")

    if answer.limitations:
        # Only show limitations when we have a substantive structured answer.
        # When JSON validation fails, `limitations` is often an internal failure hint
        # that should not leak into the user-visible chat reply.
        if answer.summary_bullets or answer.key_quotes:
            if lines:
                lines.append("")
            lines.append(f"Limitations: {answer.limitations.strip()}")
        else:
            return "Sorry—something went wrong while generating your answer. Please try again."

    if not lines:
        return "I don’t have sufficient support in the provided materials to answer that question."

    return "\n".join(lines).strip()


def generate_hyde_query(
    question: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    tenant: str,
    agent: str,
    user: str,
) -> str:
    """Generate a hypothetical relevant excerpt to improve retrieval."""

    system = (
        "You write a hypothetical excerpt that would likely appear in the relevant document.\n"
        "Rules:\n"
        "- DO NOT answer the question.\n"
        "- DO NOT add facts not implied by the question.\n"
        "- Write 8-14 lines of a plausible deposition/transcript-style excerpt.\n"
        "- No citations.\n"
        "- Output ONLY the excerpt text.\n"
    )
    rsp = get_llm_response(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tenant=tenant,
        agent=agent,
        user=user,
        question=question,
        description="hyde",
        stage="hyde",
        optional=True,
    )
    hyde = (rsp.get("content") or "").strip()
    if rsp.get("error"):
        return question
    # Conservative fallback: validate quality
    if len(hyde) < 40:
        logger.warning("HyDE generation too short (<40 chars), using original question")
        return question
    # Additional check: ensure it's not just a copy of the question
    # (simple heuristic: if identical ignoring case/whitespace, it's not useful)
    if hyde.lower().strip() == question.lower().strip():
        logger.warning("HyDE generation identical to original question, using original")
        return question
    return hyde


def _answer_system_prompt(*, evidence: list[EvidenceItem], language: str) -> str:
    evidence_lines: list[str] = []
    for ev in evidence:
        pl = _format_pl(ev)
        pl_part = f" ({pl})" if pl else ""
        heading_part = f" - {ev.heading}" if ev.heading else ""
        evidence_lines.append(
            f"{ev.citation_id}: {ev.source}{pl_part}{heading_part}\n<<<\n{ev.quote}\n>>>"
        )

    return (
        "You are a legal-tech assistant operating in a strict evidence-only mode.\n"
        "You MUST use only the Evidence items provided. Do not use prior knowledge.\n"
        "Every bullet MUST be supported by exactly one Evidence item.\n"
        "If you cannot support a point with the Evidence, omit it.\n\n"
        f"Respond in {language}.\n\n"
        "Output STRICT JSON only, matching this schema:\n"
        "{\n"
        '  "summary_bullets": [{"text": "...", "citation_id": "C1"}],\n'
        '  "key_quotes": [{"quote": "...", "citation_id": "C1"}],\n'
        '  "limitations": "..." | null\n'
        "}\n\n"
        "Style requirements:\n"
        "- summary_bullets: 5-8 bullets, concise, high-signal.\n"
        "- key_quotes: 3-6 items, short direct quotes copied from Evidence (no paraphrase).\n"
        "- Aim for ~1:5–1:6 compression vs the Evidence.\n\n"
        "Evidence:\n"
        + "\n\n".join(evidence_lines)
    )


def generate_structured_answer(
    *,
    question: str,
    evidence: list[EvidenceItem],
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    tenant: str,
    agent: str,
    user: str,
    language: str,
) -> tuple[StructuredAnswer, dict[str, Any]]:
    """Generate and validate structured JSON from the model. One repair attempt."""

    system = _answer_system_prompt(evidence=evidence, language=language)
    base_messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    rsp = get_llm_response(
        messages=base_messages,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tenant=tenant,
        agent=agent,
        user=user,
        question=question,
        description="rag_answer_json",
        stage="rag_answer_json",
    )
    raw = (rsp.get("content") or "").strip()

    def _parse(s: str) -> StructuredAnswer:
        data = json.loads(s)
        return StructuredAnswer.model_validate(data)

    # Provider failures should not enter the JSON validation/repair loop.
    if rsp.get("error"):
        return StructuredAnswer(limitations="Temporary system issue generating a structured answer."), {
            "raw": raw,
            "llm": rsp,
            "error": "llm_error",
        }

    try:
        return _parse(raw), {"raw": raw, "llm": rsp}
    except (json.JSONDecodeError, ValidationError):
        # Repair once
        repair_system = (
            "You fix JSON to match the required schema exactly. Output only valid JSON.\n"
            "Do not add new facts. Preserve the same citation_ids.\n"
        )
        repair = get_llm_response(
            messages=[
                {"role": "system", "content": repair_system},
                {"role": "user", "content": raw},
            ],
            provider=provider,
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            tenant=tenant,
            agent=agent,
            user=user,
            question=question,
            description="rag_answer_json_repair",
            stage="rag_answer_json_repair",
        )
        repaired = (repair.get("content") or "").strip()
        if repair.get("error"):
            return StructuredAnswer(limitations="Temporary system issue generating a structured answer."), {
                "raw": raw,
                "llm": rsp,
                "repair": repair,
                "error": "llm_error_repair",
            }
        try:
            return _parse(repaired), {"raw": repaired, "llm": rsp, "repair": repair}
        except Exception:
            # Last resort: safe empty answer
            return StructuredAnswer(limitations="Unable to produce a validated structured answer."), {
                "raw": raw,
                "llm": rsp,
                "repair": repair,
                "error": "validation_failed",
            }


def run_legal_rag(
    *,
    tenant: str,
    agent: str,
    question: str,
    user: str,
    language: str = "English",
    # HyDE config
    hyde_enabled: bool = True,
    hyde_provider: str = "anthropic",
    hyde_model: str = "claude-3-5-sonnet-20240620",
    hyde_temperature: float = 0.2,
    hyde_max_tokens: int | None = 400,
    # Retrieval config
    retrieval_mode: Literal["mmr", "similarity"] = "mmr",
    mmr_lambda_mult: float = 0.6,
    mmr_fetch_k: int = 50,
    final_k: int = 8,
    # Answer config
    answer_provider: str = "openai",
    answer_model: str = "gpt-4o-mini",
    answer_temperature: float = 0.2,
    answer_max_tokens: int | None = 800,
) -> RAGResult:
    """End-to-end retrieval + structured answering suitable for UI citations."""

    q_for_retrieval = question
    if hyde_enabled:
        try:
            q_for_retrieval = generate_hyde_query(
                question,
                provider=hyde_provider,
                model=hyde_model,
                temperature=hyde_temperature,
                max_tokens=hyde_max_tokens,
                tenant=tenant,
                agent=agent,
                user=user,
            )
        except Exception:
            logger.exception("HyDE failed; falling back to original question")
            q_for_retrieval = question

    # Retrieve
    chunks: list[tuple[str, dict[str, Any]]] = []
    if retrieval_mode == "mmr" and retrieve_documents_mmr is not None:
        docs = retrieve_documents_mmr(
            tenant,
            agent,
            q_for_retrieval,
            k=final_k,
            fetch_k=mmr_fetch_k,
            lambda_mult=mmr_lambda_mult,
        )
        chunks = [(d.page_content, d.metadata) for d in docs]
    else:
        sims = search_documents(tenant, agent, q_for_retrieval, k=final_k)
        chunks = [(c, m) for (c, m, _score) in sims]

    evidence = build_evidence_pack(chunks, max_items=final_k)
    evidence_by_id = {e.citation_id: e for e in evidence}

    if not evidence:
        return RAGResult(reply="No documents are available for this matter yet. Please upload files first.")

    structured, debug = generate_structured_answer(
        question=question,
        evidence=evidence,
        provider=answer_provider,
        model=answer_model,
        temperature=answer_temperature,
        max_tokens=answer_max_tokens,
        tenant=tenant,
        agent=agent,
        user=user,
        language=language,
    )
    reply = render_answer_with_citations(structured, evidence_by_id)

    # Keep backward-compatible `sources` while also returning full `evidence` for sidebar quotes.
    sources: list[dict[str, Any]] = []
    for ev in evidence:
        s: dict[str, Any] = {"source": ev.source}
        if ev.page is not None:
            s["page"] = ev.page
        if ev.line_start is not None:
            s["line"] = ev.line_start
        if ev.heading is not None:
            s["heading"] = ev.heading
        sources.append(s)

    evidence_payload = [
        {
            "citation_id": ev.citation_id,
            "source": ev.source,
            "page": ev.page,
            "line_start": ev.line_start,
            "line_end": ev.line_end,
            "heading": ev.heading,
            "quote": ev.quote,
        }
        for ev in evidence
    ]

    return RAGResult(
        reply=reply,
        sources=sources,
        evidence=evidence_payload,
        answer_json=StructuredAnswer.model_dump(structured),
    )

