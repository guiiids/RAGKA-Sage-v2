import pytest
import sys
from pathlib import Path
import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out heavy service modules before importing rag_assistant
session_memory_stub = types.ModuleType("session_memory")
session_memory_stub.SessionMemory = object
session_memory_stub.PostgresSessionMemory = object
session_memory_stub.RedisSessionMemory = object
sys.modules['services.session_memory'] = session_memory_stub

citation_service_stub = types.ModuleType("postgres_citation_service")
citation_service_stub.postgres_citation_service = types.SimpleNamespace(register_sources=lambda *a, **k: [])
sys.modules['services.postgres_citation_service'] = citation_service_stub

openai_service_stub = types.ModuleType("openai_service")
openai_service_stub.OpenAIService = type("OpenAIService", (), {})
sys.modules['openai_service'] = openai_service_stub

azure_search_stub = types.ModuleType("azure.search.documents")
azure_models_stub = types.ModuleType("azure.search.documents.models")
azure_core_stub = types.ModuleType("azure.core.credentials")
azure_search_stub.SearchClient = object
azure_models_stub.VectorizedQuery = object
azure_core_stub.AzureKeyCredential = object
sys.modules['azure'] = types.ModuleType("azure")
sys.modules['azure.search'] = types.ModuleType("search")
sys.modules['azure.search.documents'] = azure_search_stub
sys.modules['azure.search.documents.models'] = azure_models_stub
sys.modules['azure.core'] = types.ModuleType("core")
sys.modules['azure.core.credentials'] = azure_core_stub

from rag_assistant import EnhancedSimpleRedisRAGAssistant
from services.postgres_citation_service import postgres_citation_service

# Allow other tests to import the real module
sys.modules.pop('services.postgres_citation_service', None)


class DummyMemory:
    def __init__(self):
        self.data = []

    def store_turn(self, session_id, user_msg, bot_msg, summary=None):
        self.data.append((user_msg, bot_msg))

    def get_history(self, session_id, last_n_turns=10):
        return self.data[-last_n_turns:]

    def clear(self, session_id):
        self.data = []

    def get_stats(self):
        return {}


class DummyOpenAIService:
    def __init__(self, answer):
        self.answer = answer

    def get_chat_response(self, messages, max_completion_tokens=900):
        return self.answer

    def summarize_text(self, text):
        return "summary"


class DummyAssistant(EnhancedSimpleRedisRAGAssistant):
    def __init__(self, answer):
        self.session_id = "s"
        self.max_history = 5
        self.memory = DummyMemory()
        self.openai_svc = DummyOpenAIService(answer)


def test_generate_response_filters_and_renumbers(monkeypatch):
    answer = "Info a [2]. More from c [4]."
    assistant = DummyAssistant(answer)

    kb = [
        {"title": "A", "chunk": "a", "parent_id": "p1"},
        {"title": "B", "chunk": "b", "parent_id": "p2"},
        {"title": "C", "chunk": "c", "parent_id": "p3"},
        {"title": "D", "chunk": "d", "parent_id": "p4"},
    ]
    monkeypatch.setattr(assistant, "_search_kb", lambda q: kb)

    def fake_register(session_id, sources):
        for i, s in enumerate(sources, 1):
            s["citation_id"] = i
            s["display_id"] = str(i)
        return sources

    monkeypatch.setattr(postgres_citation_service, "register_sources", fake_register)

    resp, sources = assistant.generate_response("q")

    assert resp == "Info a [1]. More from c [2]."
    assert [s["citation_id"] for s in sources] == [1, 2]
    assert [s["title"] for s in sources] == ["B", "D"]
