# Implementation Plan: Making All System Prompts Configurable

## Goal
Make all system prompts and context configurable through the "Configure Agent" section in the admin UI, replacing all hard-coded prompts.

## Current State Analysis

### ✅ Already Configurable (with UI)
1. **Question Evaluator** - Has UI field `qeSystemPrompt`, uses `question_evaluator.system_prompt` from config
2. **Answer Evaluator** - Has UI field `aeSystemPrompt`, uses `answer_evaluator.system_prompt` from config

### ❌ Not Configurable (Hard-Coded)
1. **HyDE Query Generation** - Hard-coded in `rag_pipeline.py:212-220`
2. **Main RAG Answer Generation** - Hard-coded in `rag_pipeline.py:250-278`, but UI field exists (`mainSystemPrompt`) and is saved to config but **NOT USED**
3. **JSON Repair** - Hard-coded in `rag_pipeline.py:329-332`

## Implementation Plan

### Phase 1: Main RAG Answer Generation System Prompt
**Priority: HIGH** (Most critical stage, UI field already exists)

**Changes Needed:**
1. **`rag_pipeline.py`**:
   - Modify `_answer_system_prompt()` to accept optional `base_prompt` parameter
   - If `base_prompt` provided, use it as the base; otherwise use current hard-coded default
   - Keep the evidence formatting and JSON schema requirements (these are structural, not domain-specific)
   - Modify `generate_structured_answer()` to accept `base_system_prompt` parameter
   - Pass it to `_answer_system_prompt()`

2. **`rag_pipeline.py`** (`run_legal_rag()`):
   - Add `answer_system_prompt` parameter
   - Pass it to `generate_structured_answer()`

3. **`routers/chat_routes.py`**:
   - Extract `main_rag.system_prompt` from config
   - Pass it to `run_legal_rag()` as `answer_system_prompt`

**Default Behavior:**
- If no custom prompt provided, use current hard-coded default (backward compatible)
- The prompt should allow placeholders like `{language}` and `{evidence}` or we append evidence to the custom prompt

**UI:**
- Already exists (`mainSystemPrompt` field)
- Just needs to be wired up to actually work

---

### Phase 2: HyDE System Prompt
**Priority: HIGH** (Currently hard-coded, no UI exists)

**Changes Needed:**
1. **`config.py`**:
   - Add `system_prompt` to HyDE defaults in `_ensure_stage_defaults()`
   - Default to current hard-coded prompt

2. **`rag_pipeline.py`**:
   - Modify `generate_hyde_query()` to accept `system_prompt` parameter
   - Use provided prompt or fall back to current hard-coded default

3. **`rag_pipeline.py`** (`run_legal_rag()`):
   - Add `hyde_system_prompt` parameter
   - Pass it to `generate_hyde_query()`

4. **`routers/chat_routes.py`**:
   - Extract `hyde.system_prompt` from config
   - Pass it to `run_legal_rag()` as `hyde_system_prompt`

5. **`static/admin.html`**:
   - Add HyDE configuration section (can be a collapsible subsection under Main RAG or separate card)
   - Add fields: `hydeSystemPrompt` textarea
   - Update `populateAgentConfig()` to load `hyde.system_prompt`
   - Update `handleSaveAgentConfig()` to save `hyde.system_prompt`

**UI Design:**
- Add as a collapsible section within Main RAG Bot card, or as a separate small card
- Label: "HyDE Query Enhancement" (optional)
- Include the system prompt textarea

---

### Phase 3: JSON Repair System Prompt
**Priority: MEDIUM** (Less critical, but should be configurable)

**Changes Needed:**
1. **`config.py`**:
   - Add `json_repair_prompt` to `main_rag` defaults
   - Default to current hard-coded prompt

2. **`rag_pipeline.py`**:
   - Modify `generate_structured_answer()` to accept `json_repair_prompt` parameter
   - Use provided prompt or fall back to current hard-coded default

3. **`rag_pipeline.py`** (`run_legal_rag()`):
   - Add `json_repair_prompt` parameter
   - Pass it to `generate_structured_answer()`

4. **`routers/chat_routes.py`**:
   - Extract `main_rag.json_repair_prompt` from config
   - Pass it to `run_legal_rag()`

5. **`static/admin.html`**:
   - Add field in Main RAG Bot section: `mainJsonRepairPrompt` textarea
   - Update `populateAgentConfig()` to load it
   - Update `handleSaveAgentConfig()` to save it

**UI Design:**
- Add as a small textarea in the Main RAG Bot card
- Label: "JSON Repair Prompt" (optional, advanced)
- Can be collapsed/hidden by default with "Show Advanced" toggle

---

### Phase 4: Question Evaluator Default Prompt
**Priority: LOW** (Already configurable, but default is hard-coded)

**Changes Needed:**
1. **`config.py`**:
   - Make `DEFAULT_QUESTION_EVALUATOR_PROMPT` more generic or remove domain-specific language
   - Or: Keep it but document that it's a default that should be customized

**Note:** This is less critical since it's already configurable. The hard-coded default is just a starting point.

---

## Implementation Order

1. **Phase 1** (Main RAG) - Highest priority, UI already exists
2. **Phase 2** (HyDE) - High priority, needs UI addition
3. **Phase 3** (JSON Repair) - Medium priority, needs UI addition
4. **Phase 4** (Question Evaluator default) - Low priority, optional improvement

---

## Technical Details

### Main RAG Prompt Structure
The current prompt has two parts:
1. **Base instructions** (domain-specific, should be configurable):
   - "You are a legal-tech assistant..." → Should be configurable
   - "You MUST use only the Evidence items..." → Should be configurable
   
2. **Structural requirements** (should remain, but can be appended):
   - Language instruction
   - JSON schema
   - Style requirements
   - Evidence formatting

**Approach:** Allow custom base prompt, then append the structural requirements. Or use a template with placeholders.

### Backward Compatibility
- All changes must maintain backward compatibility
- If config field is missing/empty, use current hard-coded defaults
- Existing configs should continue to work without modification

### Default Prompts
When implementing, we'll:
1. Keep current hard-coded prompts as defaults
2. Allow them to be overridden via config
3. Document the default prompts in code comments

---

## Testing Plan

1. **Test with empty config** - Should use hard-coded defaults
2. **Test with custom prompts** - Should use custom prompts
3. **Test with partial config** - Should use defaults for missing fields
4. **Test UI** - Verify all fields save/load correctly
5. **Test backward compatibility** - Old configs should still work

---

## Files to Modify

### Python Files:
- `rag_pipeline.py` - Main changes for prompt parameters
- `routers/chat_routes.py` - Extract config and pass to pipeline
- `config.py` - Add default prompts to config structure

### UI Files:
- `static/admin.html` - Add UI fields for HyDE and JSON repair prompts

### Documentation:
- Update `LLM_CONTEXT_ANALYSIS.md` to reflect changes
- Add comments in code explaining prompt structure

