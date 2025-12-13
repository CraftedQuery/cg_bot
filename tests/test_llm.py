import sys
import types


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

    # Import the real module (do not globally monkeypatch `openai`, since
    # other tests depend on langchain_openai importing the OpenAI SDK).
    import cg_bot.llm as llm

    messages = [
        {"role": "system", "content": "first"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "second"},
        {"role": "assistant", "content": "there"},
    ]

    rsp = llm._get_anthropic_response(messages, model="test-model", temperature=0.1)

    assert rsp["content"] == "done"
    assert recorder['kwargs']['system'] == "first\nsecond"
    assert all(m['role'] != 'system' for m in recorder['kwargs']['messages'])
