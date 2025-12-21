# System Prompts Configuration Guide

This guide explains how to configure the system prompts (context) for each LLM call in the RAG pipeline. All system prompts are now fully configurable through the Admin UI.

## Overview

The RAG pipeline consists of multiple stages, each of which may make LLM API calls. Each stage has its own configurable system prompt that defines the context and behavior for that specific LLM interaction.

## Accessing the Configuration

1. Open the **Admin UI** in your browser
2. Navigate to the **"Configure Agent"** section
3. Select your **tenant** and **agent**
4. The configuration modal will display all configurable stages

---

## Configuration Locations by Stage

### 1. Question Evaluator (Step 1)

**Location:** First pipeline card in the configuration modal

**What it controls:**
- Pre-processes incoming user questions before retrieval
- Determines if questions should be answered, rejected, or suggested for revision
- Acts as a guardrail to filter inappropriate or out-of-scope questions

**How to configure:**
- Enable the "Question Evaluator" toggle switch
- Find the **"System Prompt"** textarea field in the Question Evaluator section
- Enter your custom prompt (or leave empty to use the default)

**Default behavior:** If left empty, uses a generic evaluation prompt that checks for:
- Scope (within knowledge base and permitted topics)
- Restricted information (confidential, PII, privileged data)
- Services outside authority (legal advice, medical advice, official decisions)
- Clarity and specificity

**Example custom prompt:**
```
You are a question evaluator for a customer support chatbot.
Evaluate if the question is:
1. Within the scope of our product documentation
2. Not requesting sensitive account information
3. Clear enough to provide a helpful answer

Respond with JSON: {"status": "pass"|"reject"|"suggest", "proceed": true|false, ...}
```

---

### 2. Main RAG Bot (Step 2) - Answer Generation

**Location:** Second pipeline card labeled "Main RAG Bot"

**What it controls:**
- The primary answer generation system prompt
- Defines the assistant's role, personality, and behavior
- **Note:** Structural requirements (JSON schema, evidence formatting) are automatically appended to your custom prompt

**How to configure:**
- Find the **"System Prompt"** textarea (labeled "System Prompt")
- Enter your custom base prompt
- The system will automatically append:
  - Language instruction
  - JSON schema requirements
  - Evidence formatting instructions
  - Style guidelines

**Important:** Your custom prompt is the **base** - you define the assistant's role and behavior. The system adds the technical requirements automatically.

**Example custom prompts:**

*Customer Support:*
```
You are a helpful customer support assistant. You provide clear, friendly, and accurate answers based on the provided documentation.
```

*Technical Documentation:*
```
You are a technical documentation assistant. You explain complex concepts clearly and provide accurate technical information from the provided materials.
```

*Legal Assistant:*
```
You are a legal-tech assistant operating in a strict evidence-only mode. You MUST use only the Evidence items provided. Do not use prior knowledge.
```

**Default behavior:** If left empty, uses: "You are a legal-tech assistant operating in a strict evidence-only mode..."

---

### 3. HyDE Query Enhancement (Step 2a) - Optional

**Location:** Separate card after Main RAG Bot, labeled "HyDE Query Enhancement"

**What it controls:**
- Generates hypothetical excerpts to improve retrieval
- Used to expand/rewrite queries before vector search
- Helps find relevant documents even when the original question is phrased differently

**How to configure:**
- Enable the "HyDE" toggle switch
- Find the **"System Prompt"** textarea in the HyDE section
- Customize how hypothetical excerpts are generated

**Default behavior:** If left empty, uses:
```
You write a hypothetical excerpt that would likely appear in the relevant document.
Rules:
- DO NOT answer the question.
- DO NOT add facts not implied by the question.
- Write 8-14 lines of a plausible excerpt that would contain relevant information.
- No citations.
- Output ONLY the excerpt text.
```

**Example custom prompt:**
```
Generate a hypothetical document excerpt that would contain the answer to this question.
The excerpt should be written in a professional, technical style matching our documentation.
Write 10-15 lines. Do not answer the question directly - just create a plausible excerpt.
```

---

### 4. JSON Repair Prompt (Advanced)

**Location:** Main RAG Bot card → "Advanced Options" (collapsible section)

**What it controls:**
- Used when the initial JSON response is invalid and needs repair
- Instructs the model to fix JSON formatting without changing content or meaning
- Only triggered if the first response fails JSON validation

**How to configure:**
- In the Main RAG Bot card, expand the **"Advanced Options"** section
- Find the **"JSON Repair Prompt"** textarea
- Customize the repair instructions

**Default behavior:** If left empty, uses:
```
You fix JSON to match the required schema exactly. Output only valid JSON.
Do not add new facts. Preserve the same citation_ids.
```

**When to customize:** Usually not necessary unless you have specific requirements for how JSON should be repaired.

---

### 5. Answer Evaluator (Step 3)

**Location:** Third pipeline card labeled "Answer Evaluator"

**What it controls:**
- Post-processing evaluation of generated answers
- Used for quality checks, flagging issues, or scoring responses
- Can be used for auditing, quality assurance, or feedback loops

**How to configure:**
- Enable the "Answer Evaluator" toggle switch
- Find the **"System Prompt"** textarea in the Answer Evaluator section
- Enter your evaluation criteria and instructions

**Default behavior:** No default prompt (field is empty by default)

**Example custom prompts:**

*Quality Check:*
```
Evaluate the following answer for:
1. Accuracy - does it correctly use the provided evidence?
2. Completeness - does it address the question fully?
3. Clarity - is it easy to understand?
4. Citations - are all claims properly cited?

Respond with JSON: {"score": 1-10, "issues": [...], "recommendations": [...]}
```

*Safety Check:*
```
Review this answer for potential issues:
- Hallucinations or unsupported claims
- Inappropriate content
- Missing citations
- Incomplete responses

Flag any concerns with JSON: {"flagged": true|false, "reason": "...", "severity": "low|medium|high"}
```

---

## Visual Guide

```
Configure Agent Modal
│
├── [Step 1] Question Evaluator
│   ├── Enabled: [toggle]
│   ├── AI Provider: [dropdown]
│   ├── Model: [dropdown]
│   └── System Prompt: [textarea] ← Configure here
│
├── [Step 2] Main RAG Bot
│   ├── Enabled: [toggle]
│   ├── AI Provider: [dropdown]
│   ├── Model: [dropdown]
│   ├── System Prompt: [textarea] ← Main answer generation context
│   └── Advanced Options (collapsible)
│       └── JSON Repair Prompt: [textarea] ← JSON repair context
│
├── [Step 2a] HyDE Query Enhancement
│   ├── Enabled: [toggle]
│   ├── AI Provider: [dropdown]
│   ├── Model: [dropdown]
│   └── System Prompt: [textarea] ← Query expansion context
│
└── [Step 3] Answer Evaluator
    ├── Enabled: [toggle]
    ├── AI Provider: [dropdown]
    ├── Model: [dropdown]
    └── System Prompt: [textarea] ← Answer evaluation context
```

---

## Important Notes

### Main RAG System Prompt Structure

The Main RAG system prompt you configure is the **base prompt**. The system automatically appends:

1. **Language instruction** - "Respond in {language}"
2. **JSON schema requirements** - The exact schema format required
3. **Evidence formatting** - How to format and cite evidence items
4. **Style guidelines** - Bullet points, quotes, compression ratios

**Example of what gets appended:**
```
[Your custom prompt here]

Respond in English.

Output STRICT JSON only, matching this schema:
{
  "summary_bullets": [{"text": "...", "citation_id": "C1"}],
  "key_quotes": [{"quote": "...", "citation_id": "C1"}],
  "limitations": "..." | null
}

Style requirements:
- summary_bullets: 5-8 bullets, concise, high-signal.
- key_quotes: 3-6 items, short direct quotes copied from Evidence (no paraphrase).
- Aim for ~1:5–1:6 compression vs the Evidence.

Evidence:
C1: document.pdf (Page 5, Line 10-15)
<<<
[Evidence content here]
>>>
```

### Backward Compatibility

- If you leave any prompt field **empty**, the system uses sensible defaults
- Existing configurations will continue to work without modification
- Defaults are designed to work well for most use cases
- **Exception:** Answer Evaluator has no default (field starts empty)

### Saving Configuration

- Click individual **"Save"** buttons for each section to save that stage
- Or use the main form submit to save all changes at once
- Changes take effect immediately for new chat requests

### Testing Your Prompts

After configuring system prompts:

1. **Test with sample questions** to verify behavior
2. **Check the logs** (Admin UI → Analytics) to see what prompts were used
3. **Review responses** to ensure they match your expectations
4. **Iterate** - adjust prompts based on actual performance

---

## Quick Reference

| Stage | Location | Default | Required? |
|-------|----------|---------|-----------|
| **Question Evaluator** | Step 1 card | Generic evaluation prompt | No (optional stage) |
| **Main RAG** | Step 2 card | "legal-tech assistant" prompt | Yes (main stage) |
| **HyDE** | Step 2a card | Hypothetical excerpt prompt | No (optional stage) |
| **JSON Repair** | Step 2 → Advanced | Simple repair prompt | No (auto-triggered) |
| **Answer Evaluator** | Step 3 card | None | No (optional stage) |

---

## Best Practices

### 1. Start with Defaults
- Begin with default prompts and test
- Only customize when you have specific requirements
- Defaults are designed to work well for most use cases

### 2. Be Specific
- Clearly define the assistant's role and domain
- Specify any constraints or requirements
- Include examples if helpful

### 3. Keep It Focused
- Each prompt should have a single, clear purpose
- Don't mix evaluation criteria with generation instructions
- Use separate stages for different concerns

### 4. Test Thoroughly
- Test with various question types
- Verify citations are correct
- Check that responses match your domain
- Review logs to see what prompts were actually used

### 5. Iterate Based on Results
- Monitor answer quality
- Adjust prompts based on actual performance
- Use Answer Evaluator to get feedback on quality
- Refine prompts over time

---

## Troubleshooting

### Prompt Not Taking Effect
- **Check:** Did you save the configuration? (Click the "Save" button)
- **Check:** Is the stage enabled? (Toggle switch must be ON)
- **Check:** Are you testing with a new chat request? (Changes apply to new requests only)

### Unexpected Behavior
- **Check:** Review the actual prompt used in the logs (Admin UI → Analytics → LLM Events)
- **Check:** Verify your prompt syntax and formatting
- **Check:** Ensure the prompt matches the expected JSON format (for evaluators)

### Default Prompt Still Being Used
- **Check:** Did you leave the field empty? (Empty = uses default)
- **Check:** Did you save the configuration?
- **Check:** Is there whitespace-only content? (May be treated as empty)

---

## Related Documentation

- [RAG Request Flow](./rag-request-flow.md) - Detailed explanation of the pipeline stages
- [LLM Context Analysis](../LLM_CONTEXT_ANALYSIS.md) - Technical analysis of where context is set
- [README](../README.md) - General project documentation

---

## Support

For issues or questions about system prompt configuration:
1. Check the logs in Admin UI → Analytics
2. Review the [RAG Request Flow](./rag-request-flow.md) documentation
3. Test with simple prompts first, then add complexity
4. Verify your JSON format matches the expected schema (for evaluators)

