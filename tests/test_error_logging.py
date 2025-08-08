import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Ensure package parent on path and set data directory
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root.parent))
    monkeypatch.setenv("RAG_CHATBOT_HOME", str(tmp_path))

    lang_mod = types.ModuleType("langdetect")
    lang_mod.detect = lambda _text: "en"
    lang_mod.DetectorFactory = types.SimpleNamespace(seed=0)
    sys.modules["langdetect"] = lang_mod

    # Stub external auth dependencies
    jose_mod = types.ModuleType("jose")
    jose_mod.JWTError = Exception
    jose_mod.jwt = types.SimpleNamespace(encode=lambda *a, **k: "x", decode=lambda *a, **k: {})
    sys.modules["jose"] = jose_mod

    passlib_mod = types.ModuleType("passlib")
    context_mod = types.ModuleType("passlib.context")

    class DummyCryptContext:
        def __init__(self, *args, **kwargs):
            pass

        def hash(self, password):
            return "hashed-" + password

        def verify(self, plain, hashed):
            return hashed == "hashed-" + plain

    context_mod.CryptContext = DummyCryptContext
    passlib_mod.context = context_mod
    sys.modules["passlib"] = passlib_mod
    sys.modules["passlib.context"] = context_mod

    # Remove previously loaded project modules
    for name in list(sys.modules):
        if name.startswith("cg_bot"):
            del sys.modules[name]

    # Stub heavy modules
    ingest_stub = types.ModuleType("cg_bot.ingestion")
    ingest_stub.ingest = lambda *a, **k: {}
    sys.modules["cg_bot.ingestion"] = ingest_stub

    vector_stub = types.ModuleType("cg_bot.vectorstore")
    vector_stub.clear_cache = lambda *a, **k: None
    vector_stub.search_documents = lambda *a, **k: []
    vector_stub.update_vector_store = lambda *a, **k: None
    vector_stub.get_vector_store_size = lambda *a, **k: 0
    sys.modules["cg_bot.vectorstore"] = vector_stub

    import cg_bot.config  # noqa: F401
    import cg_bot.database as database
    database.init_database()
    import cg_bot.auth as auth
    import cg_bot.main as main

    app = main.app
    app.dependency_overrides[auth.get_files_user] = lambda: types.SimpleNamespace(
        username="admin", tenant="public", role="admin", allow_files=True, agents=[]
    )
    app.dependency_overrides[auth.get_admin_user] = lambda: types.SimpleNamespace(
        username="admin", tenant="public", role="admin", allow_files=True, agents=[]
    )

    @app.get("/boom")
    async def boom():
        raise Exception("boom")

    return TestClient(app, raise_server_exceptions=False)


def test_error_logging(client):
    import cg_bot.database as database

    files = [("files", ("test.txt", b"hello", "text/plain"))]
    r1 = client.post("/upload", files=files, params={"tenant": "public", "agent": "default"})
    assert r1.status_code == 200

    r2 = client.post("/upload", files=files, params={"tenant": "public", "agent": "default"})
    assert r2.status_code == 409

    logs = database.get_error_logs()
    assert len(logs) == 1
    ts, endpoint, tenant, agent, message = logs[0]
    assert endpoint == "/upload"
    assert tenant == "public" and agent == "default"
    assert "already exists" in message

    r3 = client.get("/boom", params={"tenant": "public", "agent": "default"})
    assert r3.status_code == 500

    logs = database.get_error_logs(limit=2)
    assert len(logs) == 2
    (_, ep1, t1, a1, msg1), (_, ep2, t2, a2, msg2) = logs
    assert ep1 == "/boom" and msg1 == "boom"
    assert t1 == "public" and a1 == "default"
    assert ep2 == "/upload"

    resp = client.get("/error_logs", params={"limit": 2})
    assert resp.status_code == 200
    data = resp.json()["logs"]
    assert data[0]["endpoint"] == "/boom"
    assert data[1]["endpoint"] == "/upload"


def test_error_logs_requires_authentication(client):
    import cg_bot.auth as auth

    # Remove admin override so auth dependencies are enforced
    client.app.dependency_overrides.pop(auth.get_files_user, None)
    client.app.dependency_overrides.pop(auth.get_admin_user, None)

    resp = client.get("/error_logs")
    assert resp.status_code == 401
