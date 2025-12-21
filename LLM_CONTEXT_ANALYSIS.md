# LLM Context Analysis - System Prompt Mapping

This document analyzes where system context (system prompts) is set for each LLM API call in the RAG workflow.

**📖 User Guide:** For instructions on how to configure these prompts through the Admin UI, see [`docs/system-prompts-configuration.md`](docs/system-prompts-configuration.md).

**Note:** This is a technical analysis document. All prompts described here are now fully configurable through the Admin UI.

## Workflow Overview

The RAG pipeline consists of the following stages that may make LLM API calls:

1. **Question Evaluator** (optional)
2. **HyDE Query Generation** (optional)
3. **Retrieval** (embedding call, not LLM)
4. **Answer Generation** (main RAG)
5. **JSON Repair** (optional, if answer generation fails validation)
6. **Answer Evaluator** (optional)

---

## 1. Question Evaluator Stage

**Location**: `routers/chat_routes.py` (lines 136-158)

**System Context Source**: 
- **Configurable** via `question_evaluator.system_prompt` in tenant/agent config file
- **Default fallback**: `DEFAULT_QUESTION_EVALUATOR_PROMPT` in `config.py` (lines 22-61)

**Code Reference**:
```python
# routers/chat_routes.py:140-141
if qe_cfg.get("system_prompt"):
    qe_messages.append({"role": "system", "content": qe_cfg.get("system_prompt", "")})
```

**Default Prompt** (from `config.py:22-61`):
```
You are ONLY evaluating if a user question is appropriate for the municipal government chatbot. You are NOT answering the question.

Your job is to assess the question against the evaluation criteria and return a brief evaluation summary. NEVER provide information that answers the user's question.

Evaluate based on:
- Is it within scope (city services, policies, procedures, public information)?
- Does it request restricted information (confidential, PII, privileged)?
- Does it ask for services outside our authority (legal advice, medical advice, official decisions)?
- Is it clear and specific enough to answer?

Respond ONLY with JSON in one of these formats:
[...]
```

**Status**: ⚠️ **HARD-CODED DEFAULT** - While configurable, there is a hard-coded default prompt in `config.py` that is specific to "municipal government chatbot" use case.

---

## 2. HyDE Query Generation Stage

**Location**: `rag_pipeline.py` (lines 199-247)

**System Context Source**: 
- **HARD-CODED** in `generate_hyde_query()` function (lines 212-220)

**Code Reference**:
```python
# rag_pipeline.py:212-220
system = (
    "You write a hypothetical excerpt that would likely appear in the relevant document.\n"
    "Rules:\n"
    "- DO NOT answer the question.\n"
    "- DO NOT add facts not implied by the question.\n"
    "- Write 8-14 lines of a plausible deposition/transcript-style excerpt.\n"
    "- No citations.\n"
    "- Output ONLY the excerpt text.\n"
)
```

**Status**: 🔴 **FULLY HARD-CODED** - This system prompt is hard-coded in the function and cannot be configured. It assumes a "deposition/transcript-style" format which may not be appropriate for all use cases.

---

## 3. Answer Generation (Main RAG) Stage

**Location**: `rag_pipeline.py` (lines 281-366)

**System Context Source**: 
- **HARD-CODED** in `_answer_system_prompt()` function (lines 250-278)
- Note: The config has a `main_rag.system_prompt` field, but **it is NOT used** in the actual RAG pipeline

**Code Reference**:
```python
# rag_pipeline.py:250-278
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
```

**Status**: 🔴 **FULLY HARD-CODED** - The system prompt is hard-coded with:
- "legal-tech assistant" terminology (line 261)
- "strict evidence-only mode" (line 261)
- Specific JSON schema requirements
- Style requirements for bullets and quotes

**Important Note**: The `main_rag.system_prompt` field in the config file is **NOT USED** by the RAG pipeline. It exists in the config structure but is ignored during answer generation.

---

## 4. JSON Repair Stage

**Location**: `rag_pipeline.py` (lines 329-348)

**System Context Source**: 
- **HARD-CODED** in `generate_structured_answer()` function (lines 329-332)

**Code Reference**:
```python
# rag_pipeline.py:329-332
repair_system = (
    "You fix JSON to match the required schema exactly. Output only valid JSON.\n"
    "Do not add new facts. Preserve the same citation_ids.\n"
)
```

**Status**: 🔴 **FULLY HARD-CODED** - This is a simple repair prompt, but it's still hard-coded.

---

## 5. Answer Evaluator Stage

**Location**: `routers/chat_routes.py` (lines 334-356)

**System Context Source**: 
- **Configurable** via `answer_evaluator.system_prompt` in tenant/agent config file
- **No default fallback** - if not provided, no system message is sent

**Code Reference**:
```python
# routers/chat_routes.py:338-339
if ae_cfg.get("system_prompt"):
    ae_messages.append({"role": "system", "content": ae_cfg.get("system_prompt", "")})
```

**Status**: ✅ **CONFIGURABLE** - This stage properly uses configuration and has no hard-coded defaults.

---

## Summary Table

| Stage | System Prompt Source | Hard-Coded? | Configurable? | Notes |
|-------|---------------------|-------------|---------------|-------|
| **Question Evaluator** | `config.py` default + config file | ⚠️ Yes (default) | ✅ Yes | Default is hard-coded for "municipal government chatbot" |
| **HyDE** | `rag_pipeline.py:212-220` | 🔴 Yes | ❌ No | Assumes "deposition/transcript-style" format |
| **Answer Generation** | `rag_pipeline.py:250-278` | 🔴 Yes | ❌ No | "legal-tech assistant" terminology; config field exists but unused |
| **JSON Repair** | `rag_pipeline.py:329-332` | 🔴 Yes | ❌ No | Simple repair prompt |
| **Answer Evaluator** | Config file only | ✅ No | ✅ Yes | Properly configurable |

---

## Key Findings

### 🔴 Critical Issues

1. **Main RAG Answer Generation** - The most important stage has a hard-coded system prompt that:
   - Uses "legal-tech assistant" terminology (not generic)
   - References "strict evidence-only mode" 
   - Has specific JSON schema requirements
   - **Ignores the `main_rag.system_prompt` config field entirely**

2. **HyDE Stage** - Hard-coded prompt assumes "deposition/transcript-style" format, which may not fit all document types.

### ⚠️ Warning Issues

1. **Question Evaluator Default** - While configurable, the default prompt in `config.py` is hard-coded for "municipal government chatbot" use case, which may not be appropriate for all tenants/agents.

### ✅ Good Practices

1. **Answer Evaluator** - Properly uses configuration with no hard-coded defaults.

---

## Recommendations

✅ **IMPLEMENTED:** All recommendations have been implemented:

1. ✅ **Answer Generation is now configurable**: `main_rag.system_prompt` is now fully wired up and functional through the Admin UI.

2. ✅ **HyDE is now configurable**: `hyde.system_prompt` has been added to config with a generic default, and is accessible through the Admin UI.

3. ✅ **Question Evaluator default updated**: The default prompt has been made more generic (removed "municipal government" specific language).

4. ✅ **JSON Repair prompt added**: `main_rag.json_repair_prompt` is now configurable through the Admin UI (Advanced Options).

**See [`docs/system-prompts-configuration.md`](docs/system-prompts-configuration.md) for user-facing documentation on how to configure all system prompts.**

