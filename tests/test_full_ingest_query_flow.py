import json
import re


def _fake_llm_response(*, messages, provider="openai", model=None, temperature=0.0, **kwargs):
    desc = kwargs.get("description")

    if desc == "hyde":
        return {"content": "(hypothetical excerpt for retrieval)", "tokens_out": 30}

    if desc == "rag_answer_json":
        # Cite C1 for deterministic assertions.
        payload = {
            "summary_bullets": [
                {"text": "This answer is grounded in the uploaded transcript.", "citation_id": "C1"},
                {"text": "Each point ends with a structured citation.", "citation_id": "C1"},
            ],
            "key_quotes": [
                {"quote": "LINE 1: ...", "citation_id": "C1"}
            ],
            "limitations": None,
        }
        return {"content": json.dumps(payload), "tokens_out": 120}

    return {"content": "{}", "tokens_out": 5}


def test_ingest_then_query_integration(legal_client, monkeypatch):
    import cg_bot.rag_pipeline as rag_pipeline
    monkeypatch.setattr(rag_pipeline, "get_llm_response", _fake_llm_response)

    # Ingest via upload endpoint.
    transcript = "\n".join([f"LINE {i}: Sample deposition content." for i in range(1, 101)])
    files = [("files", ("fixture.txt", transcript.encode("utf-8"), "text/plain"))]

    up = legal_client.post("/upload", files=files, params={"tenant": "public", "agent": "default"})
    assert up.status_code == 200

    q = {"messages": [{"role": "user", "content": "Summarize the key points."}]}
    r = legal_client.post("/chat", params={"tenant": "public", "agent": "default"}, json=q)
    assert r.status_code == 200

    data = r.json()

    # Must include evidence sidebar payload.
    assert isinstance(data.get("evidence"), list)
    assert len(data["evidence"]) >= 1

    # Every bullet line should have [Source: ...] {cite:CX}
    reply = data.get("reply", "")
    assert re.search(r"\[Source:[^\]]+\]\s*\{cite:C\d+\}", reply)

    # Evidence ids should be resolvable
    ids = {ev["citation_id"] for ev in data["evidence"] if "citation_id" in ev}
    for m in re.findall(r"\{cite:(C\d+)\}", reply):
        assert m in ids
