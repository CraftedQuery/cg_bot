def test_upload_rejects_missing_files(legal_client):
    r = legal_client.post("/upload", params={"tenant": "public", "agent": "default"})
    # FastAPI validation error
    assert r.status_code in {400, 422}
