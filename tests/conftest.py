import os
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class DummyEmbeddings:
    """Deterministic, dependency-free embeddings for tests."""

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text: str):
        # Tiny deterministic vector based on char codes.
        s = sum(ord(c) for c in (text or ""))
        return [float((s % 97) / 97.0), float((s % 193) / 193.0), float((s % 389) / 389.0)]


@pytest.fixture()
def legal_client(tmp_path, monkeypatch):
    """FastAPI client with auth + embeddings stubbed, real vectorstore enabled."""

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root.parent))
    monkeypatch.setenv("RAG_CHATBOT_HOME", str(tmp_path))

    # Stub language detection to keep tests deterministic.
    lang_mod = types.ModuleType("langdetect")
    lang_mod.detect = lambda _text: "en"
    lang_mod.DetectorFactory = types.SimpleNamespace(seed=0)
    sys.modules["langdetect"] = lang_mod

    # Stub crypto/JWT deps (routers are tested via dependency overrides, but modules import these).
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

    # Ensure a clean import of project modules.
    for name in list(sys.modules):
        if name.startswith("cg_bot"):
            del sys.modules[name]

    import cg_bot.database as database
    database.init_database()

    import cg_bot.auth as auth
    import cg_bot.embedding as embedding
    import cg_bot.vectorstore as vectorstore
    import cg_bot.main as main

    # Deterministic embeddings so vectorstore/FAISS can run in CI.
    monkeypatch.setattr(embedding, "get_embedding_model", lambda *a, **k: DummyEmbeddings())
    monkeypatch.setattr(vectorstore, "get_embedding_model", lambda *a, **k: DummyEmbeddings())

    app = main.app

    # Bypass auth for unit/integration tests.
    app.dependency_overrides[auth.get_current_active_user] = lambda: types.SimpleNamespace(
        username="tester", tenant="public", role="admin", allow_files=True, agents=["*"]
    )
    app.dependency_overrides[auth.get_files_user] = lambda: types.SimpleNamespace(
        username="tester", tenant="public", role="admin", allow_files=True, agents=["*"]
    )
    app.dependency_overrides[auth.get_admin_user] = lambda: types.SimpleNamespace(
        username="tester", tenant="public", role="admin", allow_files=True, agents=["*"]
    )

    return TestClient(app, raise_server_exceptions=False)
