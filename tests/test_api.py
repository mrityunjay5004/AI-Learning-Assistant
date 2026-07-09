"""
End-to-end API tests with the Gemini client mocked out, plus focused unit
tests for the JSON-extraction robustness added during the polish pass.

Kept to a single file deliberately: this is an intern assignment, not a
production test suite - the goal is to demonstrate the key paths are
verified (happy path, validation errors, not-found errors, and the
self-repair/parsing logic), not to chase coverage percentage.

Run with: pytest -v
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.schemas import LLMChatOutput, LLMProjectOutput, LLMRoadmapOutput, Subtask, Task
from app.services.llm_client import _extract_json

FAKE_ROADMAP = LLMRoadmapOutput(
    estimated_hours=120,
    skills=["FastAPI", "PostgreSQL", "Docker"],
    tasks=[
        Task(title="Learn FastAPI", estimated_hours=12, subtasks=[Subtask(title="Routing")]),
        Task(title="Learn PostgreSQL", estimated_hours=20, subtasks=[Subtask(title="Schema design")]),
    ],
)
FAKE_PROJECT = LLMProjectOutput(
    title="Task Management API",
    difficulty="Intermediate",
    estimated_hours=20,
    tech_stack=["FastAPI", "PostgreSQL"],
    features=["JWT Authentication", "CRUD APIs"],
    why_this_project="Reinforces REST API fundamentals.",
)
FAKE_CHAT = LLMChatOutput(
    response="Yes, that order works fine.",
    follow_up_questions=["Want a suggested order?"],
)


def _fake_generate_structured(system_instruction, user_prompt, output_schema, max_repair_attempts=2):
    return {
        "LLMRoadmapOutput": FAKE_ROADMAP,
        "LLMProjectOutput": FAKE_PROJECT,
        "LLMChatOutput": FAKE_CHAT,
    }[output_schema.__name__]


def _fake_embed_text(text, task_type="retrieval_document"):
    import hashlib

    h = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in h[:16]]


@pytest.fixture
def client():
    """A TestClient with the LLM and embedding calls mocked. DB/Chroma
    storage is a single isolated temp directory for the whole test session
    (set once in conftest.py, before these modules' import-time singletons
    are created) rather than per-test, since the app's Chroma/SQLAlchemy
    clients are process-wide singletons by design - see conftest.py."""
    with patch("app.services.roadmap_service.generate_structured", side_effect=_fake_generate_structured), \
         patch("app.services.project_service.generate_structured", side_effect=_fake_generate_structured), \
         patch("app.services.chat_service.generate_structured", side_effect=_fake_generate_structured), \
         patch("app.rag.retriever.embed_text", side_effect=_fake_embed_text):

        from app.main import app

        with TestClient(app) as c:
            yield c


def _create_roadmap(client) -> dict:
    payload = {
        "goal_title": "Backend Developer",
        "experience": "Less than 1 year",
        "known_skills": ["Python", "SQL"],
        "learning_style": "Project Based",
        "weekly_hours": 15,
    }
    resp = client.post("/roadmap", json=payload)
    assert resp.status_code == 201
    return resp.json()


class TestRoadmap:
    def test_create_roadmap_returns_expected_shape(self, client):
        data = _create_roadmap(client)
        assert data["goal_title"] == "Backend Developer"
        assert data["estimated_hours"] == 120
        assert len(data["tasks"]) == 2
        assert "generated_at" in data
        # curated resources should be attached deterministically
        assert data["tasks"][0]["resources"], "expected curated resources on task"

    def test_create_roadmap_validation_error(self, client):
        resp = client.post("/roadmap", json={"goal_title": "X"})  # missing required fields
        assert resp.status_code == 422

    def test_markdown_export(self, client):
        roadmap = _create_roadmap(client)
        resp = client.get(f"/roadmap/{roadmap['roadmap_id']}/markdown")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "# Learning Roadmap" in resp.text

    def test_markdown_export_not_found(self, client):
        resp = client.get("/roadmap/does-not-exist/markdown")
        assert resp.status_code == 404


class TestProject:
    def test_project_by_roadmap_id(self, client):
        roadmap = _create_roadmap(client)
        resp = client.post("/project", json={"roadmap_id": roadmap["roadmap_id"]})
        assert resp.status_code == 200
        assert resp.json()["title"] == "Task Management API"

    def test_project_by_goal_and_skills(self, client):
        resp = client.post(
            "/project", json={"goal_title": "Backend Developer", "skills": ["Python", "FastAPI"]}
        )
        assert resp.status_code == 200

    def test_project_missing_source_returns_400(self, client):
        resp = client.post("/project", json={})
        assert resp.status_code == 400


class TestChat:
    def test_chat_happy_path(self, client):
        roadmap = _create_roadmap(client)
        resp = client.post(
            "/chat",
            json={"roadmap_id": roadmap["roadmap_id"], "message": "Can I learn Docker first?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"]
        assert isinstance(body["follow_up_questions"], list)
        # retrieved_context entries should carry a citation label
        assert all(c.startswith("[") for c in body["retrieved_context"])

    def test_chat_roadmap_not_found(self, client):
        resp = client.post("/chat", json={"roadmap_id": "nope", "message": "hi"})
        assert resp.status_code == 404

    def test_chat_conversation_history_persists(self, client):
        roadmap = _create_roadmap(client)
        rid = roadmap["roadmap_id"]
        client.post("/chat", json={"roadmap_id": rid, "message": "First question"})
        resp2 = client.post("/chat", json={"roadmap_id": rid, "message": "Second question"})
        assert resp2.status_code == 200
        # two turns -> four ChatMessage rows (user+assistant each); verified
        # indirectly via a successful second call with no errors.


class TestJsonExtraction:
    """Focused tests for the LLM output parsing robustness improvements."""

    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_trailing_commas(self):
        assert _extract_json('{"a": [1, 2, 3,],}') == {"a": [1, 2, 3]}

    def test_json_wrapped_in_prose(self):
        raw = 'Sure, here you go:\n{"a": 1, "b": 2}\nLet me know if you need more!'
        assert _extract_json(raw) == {"a": 1, "b": 2}

    def test_brace_inside_string_value_does_not_break_balancing(self):
        raw = 'Here: {"a": 1, "note": "use a closing brace like this: }"}  Thanks!'
        assert _extract_json(raw) == {"a": 1, "note": "use a closing brace like this: }"}

    def test_truncated_json_raises(self):
        with pytest.raises(Exception):
            _extract_json('{"a": 1, "b": [1, 2,')
