import json
import re


def _fake_llm_response(*, messages, provider="openai", model=None, temperature=0.0, **kwargs):
    desc = kwargs.get("description")

    if desc == "hyde":
        return {"content": "Q: ...\nA: ...\n(placeholder hypothetical excerpt)", "tokens_out": 50}

    if desc in {"rag_answer_json", "rag_answer_json_repair"}:
        # Always cite C1 so the test is robust to retrieval ordering.
        payload = {
            "summary_bullets": [
                {"text": "The record contains relevant testimony addressing the question.", "citation_id": "C1"},
                {"text": "The key point is supported directly by the cited excerpt.", "citation_id": "C1"},
            ],
            "key_quotes": [
                {"quote": "(direct quote copied from the record)", "citation_id": "C1"}
            ],
            "limitations": None,
        }
        return {"content": json.dumps(payload), "tokens_out": 120}

    # Default: keep other stages inert.
    return {"content": "{}", "tokens_out": 10}


def test_chat_returns_structured_citations_and_evidence(legal_client, monkeypatch, tmp_path):
    # Upload a small text fixture via API to build the vector store.
    text = "\n".join([f"LINE {i}: Troy Brown sample text." for i in range(1, 61)])

    files = [("files", ("troy_brown_sample.txt", text.encode("utf-8"), "text/plain"))]
    r_up = legal_client.post("/upload", files=files, params={"tenant": "public", "agent": "default"})
    assert r_up.status_code == 200

    # Patch LLM calls inside the RAG pipeline.
    import cg_bot.rag_pipeline as rag_pipeline

    monkeypatch.setattr(rag_pipeline, "get_llm_response", _fake_llm_response)

    payload = {"messages": [{"role": "user", "content": "What does the witness say about the timeline?"}]}
    r = legal_client.post("/chat", params={"tenant": "public", "agent": "default"}, json=payload)
    assert r.status_code == 200

    data = r.json()
    assert "reply" in data
    assert "evidence" in data
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) >= 1

    # Must include forced structured citations.
    assert re.search(r"\[Source:[^\]]+\]\s*\{cite:C\d+\}", data["reply"]) is not None

    # Evidence items must include page/line metadata keys.
    ev = data["evidence"][0]
    assert "citation_id" in ev and ev["citation_id"].startswith("C")
    assert "source" in ev
    assert "quote" in ev

    # Answer JSON present for UI debugging.
    assert "answer_json" in data
    if data["answer_json"] is not None:
        assert "summary_bullets" in data["answer_json"]


def test_chat_requires_auth_when_not_overridden(tmp_path, monkeypatch):
    # Minimal smoke: without overrides, endpoint should enforce auth.
    # We import a fresh app without dependency overrides.
    import sys
    for name in list(sys.modules):
        if name.startswith("cg_bot"):
            del sys.modules[name]

    monkeypatch.setenv("RAG_CHATBOT_HOME", str(tmp_path))

    import cg_bot.database as database
    database.init_database()
    import cg_bot.main as main

    from fastapi.testclient import TestClient

    client = TestClient(main.app, raise_server_exceptions=False)
    r = client.post("/chat", params={"tenant": "public", "agent": "default"}, json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code in {401, 403}
