import os
import json
import importlib.util
import types
from pathlib import Path

import pytest

# Temporarily set BASE_DIR to a temp directory
@pytest.fixture()
def temp_users(tmp_path, monkeypatch):
    data_dir = tmp_path
    monkeypatch.setenv("RAG_CHATBOT_HOME", str(data_dir))
    # Reload modules that depend on BASE_DIR
    import sys
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    for mod_name in ['config', 'models']:
        spec = importlib.util.spec_from_file_location(mod_name, os.path.join(base, f"{mod_name}.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[mod_name] = module

    # Provide a dummy jose module to satisfy imports
    jose_mod = types.ModuleType('jose')
    jose_mod.JWTError = Exception
    jose_mod.jwt = types.SimpleNamespace(encode=lambda *a, **k: 'x', decode=lambda *a, **k: {})
    sys.modules['jose'] = jose_mod

    passlib_mod = types.ModuleType('passlib')
    context_mod = types.ModuleType('passlib.context')

    class DummyCryptContext:
        def __init__(self, *args, **kwargs):
            pass

        def hash(self, password):
            return 'hashed-' + password

        def verify(self, plain, hashed):
            return hashed == 'hashed-' + plain

    context_mod.CryptContext = DummyCryptContext
    passlib_mod.context = context_mod
    sys.modules['passlib'] = passlib_mod
    sys.modules['passlib.context'] = context_mod

    spec_path = os.path.join(base, 'auth.py')
    spec = importlib.util.spec_from_file_location('auth', spec_path)
    auth = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(auth)
    sys.modules['auth'] = auth
    yield data_dir


def test_admin_role_migration(temp_users):
    users_path = temp_users / "users.json"
    users_path.write_text(json.dumps({"admin": {"username": "admin", "tenant": "*", "role": None, "hashed_password": "x"}}))
    import auth
    users = auth.get_users_db()
    assert users["admin"]["role"] == "system_admin"
    # file should be updated
    saved = json.loads(users_path.read_text())
    assert saved["admin"]["role"] == "system_admin"


def test_corrupted_users_db_resets_to_admin(temp_users):
    """Corrupted users.json should recreate default admin account"""
    users_path = temp_users / "users.json"
    users_path.write_text("{bad json")

    import auth

    users = auth.get_users_db()
    assert "admin" in users
    assert users["admin"]["role"] == "system_admin"
    saved = json.loads(users_path.read_text())
    assert "admin" in saved
