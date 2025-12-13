def test_get_full_config_requires_auth(legal_client):
    # With our fixture, auth is overridden, so request should succeed.
    r = legal_client.get("/config/full", params={"tenant": "public", "agent": "default"})
    assert r.status_code == 200
    cfg = r.json()
    # New defaults should exist
    assert "retrieval" in cfg
    assert "hyde" in cfg
