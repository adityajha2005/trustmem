"""API endpoint tests."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "multi-agent-memory-test"


def test_memory_write_and_query(client: TestClient) -> None:
    write = client.post(
        "/memory/write",
        json={
            "content": "Shared task: review PR #42",
            "agent": "agent_a",
            "scope": "shared",
            "task_id": "task_1",
        },
    )
    assert write.status_code == 200
    memory_id = write.json()["memory_id"]

    response = client.post(
        "/memory/query",
        json={
            "query": "review PR",
            "agent_id": "agent_b",
            "task_id": "task_1",
            "top_k": 3,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "review PR"
    assert any(h.get("memory_id") == memory_id for h in data["hits"])
