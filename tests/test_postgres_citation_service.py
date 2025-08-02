import os
import sys

import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class FakeCursor:
    """Simple cursor implementation for in-memory testing."""

    def __init__(self, connection):
        self.connection = connection
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass

    def execute(self, query, params=None):
        if query.startswith("CREATE TABLE") or query.startswith("CREATE INDEX"):
            # Table creation queries are no-ops for the fake cursor
            self._result = None
        elif "SELECT citation_id FROM session_citation_lookup" in query:
            session_id, source_hash = params
            citation_id = self.connection.lookup.get((session_id, source_hash))
            self._result = (citation_id,) if citation_id is not None else None
        elif "INSERT INTO session_citations" in query:
            session_id, citation_id, title, content, url, source_hash = params
            self.connection.citations.append(
                {
                    "session_id": session_id,
                    "citation_id": citation_id,
                    "title": title,
                    "content": content,
                    "url": url,
                    "source_hash": source_hash,
                }
            )
            self._result = None
        elif "INSERT INTO session_citation_lookup" in query:
            session_id, source_hash, citation_id = params
            self.connection.lookup[(session_id, source_hash)] = citation_id
            self._result = None
        elif "SELECT COALESCE(MAX(citation_id)" in query:
            session_id = params[0]
            max_id = max(
                [c["citation_id"] for c in self.connection.citations if c["session_id"] == session_id],
                default=0,
            )
            self._result = (max_id + 1,)
        else:
            raise NotImplementedError(f"Query not supported: {query}")

    def fetchone(self):
        return self._result


class FakeConnection:
    def __init__(self):
        self.citations = []
        self.lookup = {}

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def service(monkeypatch):
    from db_manager import DatabaseManager
    import importlib

    fake_conn = FakeConnection()
    monkeypatch.setattr(DatabaseManager, "get_connection", lambda: fake_conn)

    module = importlib.import_module("services.postgres_citation_service")
    svc = module.PostgresCitationService()
    return svc, fake_conn


def test_register_sources_assigns_id_without_unboundlocalerror(service):
    svc, fake_conn = service
    sources = [{"title": "Doc1", "content": "Content1"}]

    registered = svc.register_sources("session1", sources)

    assert registered[0]["citation_id"] == 1
    assert len(fake_conn.citations) == 1
    assert fake_conn.citations[0]["title"] == "Doc1"


def test_register_sources_stores_multiple_new_sources(service):
    svc, fake_conn = service
    sources = [
        {"title": "Doc1", "content": "Content1"},
        {"title": "Doc2", "content": "Content2"},
    ]

    registered = svc.register_sources("session1", sources)

    assert [s["citation_id"] for s in registered] == [1, 2]
    assert len(fake_conn.citations) == 2
    assert fake_conn.citations[1]["title"] == "Doc2"

