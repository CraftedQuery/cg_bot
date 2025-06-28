import os
import sys
import types
import importlib.util

import pytest

# Create a dummy database module to satisfy llm imports
dummy_db = types.ModuleType('database')
dummy_db.log_llm_event = lambda *a, **kw: None
sys.modules['database'] = dummy_db
dummy_openai = types.ModuleType('openai')
dummy_openai.OpenAI = object
sys.modules['openai'] = dummy_openai

spec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm.py'))
spec = importlib.util.spec_from_file_location('llm', spec_path)
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)
_get_anthropic_response = llm._get_anthropic_response


class DummyMessages:
    def __init__(self, recorder):
        self.recorder = recorder

    def create(self, **kwargs):
        # record arguments passed for assertions
        self.recorder['kwargs'] = kwargs
        return types.SimpleNamespace(content=[types.SimpleNamespace(text="done")])


class DummyAnthropic:
    def __init__(self, api_key=None):
        self.messages = DummyMessages(recorder)


recorder = {}


def test_anthropic_multiple_system(monkeypatch):
    monkeypatch.setitem(sys.modules, 'anthropic', types.SimpleNamespace(Anthropic=DummyAnthropic))
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')

    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "second"},
        {"role": "assistant", "content": "there"},
    ]

    rsp = _get_anthropic_response(messages, model="test-model", temperature=0.1)

    assert rsp["content"] == "done"
    assert recorder['kwargs']['system'] == "first\nsecond"
    assert all(m['role'] != 'system' for m in recorder['kwargs']['messages'])
