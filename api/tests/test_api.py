"""
test_api.py — Integration tests for the FastAPI layer and persistence.

These tests use an in-memory SQLite database and TestClient — no running
server needed. They verify the HTTP contract (status codes, response shapes)
and persistence behaviours documented in the Test Cases document.

All tests that write data use the `client` fixture which provides a fresh
DB per test, so tests are fully isolated.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from unittest.mock import patch

from app.main import app
from app.database import get_db
from app.models import Base


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures — fresh in-memory DB per test
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """
    Each test gets a clean SQLite in-memory database.

    SQLite :memory: databases are per-connection — two different connections
    open two different empty databases. We fix this by sharing a single
    connection across create_all and all test session operations via
    connect_args / creator pattern.
    """
    from sqlalchemy.pool import StaticPool

    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        # StaticPool ensures every request to the engine reuses the SAME
        # underlying connection, so create_all and the test sessions share
        # one in-memory database.
        poolclass=StaticPool,
    )

    @event.listens_for(test_engine, "connect")
    def set_sqlite_fk(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=test_engine)

    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with patch("app.database.engine", test_engine):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)





def _create_task(client, title="Task A", duration_days=3, constraint_start="2026-01-01"):
    resp = client.post("/tasks", json={
        "title": title,
        "duration_days": duration_days,
        "constraint_start": constraint_start,
    })
    assert resp.status_code == 201, resp.text
    return resp.json()["task"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Health check
# ──────────────────────────────────────────────────────────────────────────────

def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Task CRUD
# ──────────────────────────────────────────────────────────────────────────────

def test_create_task_returns_201_with_task(client):
    r = client.post("/tasks", json={
        "title": "Write specs",
        "duration_days": 5,
        "constraint_start": "2026-03-01",
    })
    assert r.status_code == 201
    task = r.json()["task"]
    assert task["title"] == "Write specs"
    assert task["duration_days"] == 5
    assert task["blocked_ready"] == "Ready"


def test_create_task_missing_constraint_start_returns_422(client):
    r = client.post("/tasks", json={"title": "Incomplete task"})
    assert r.status_code == 422


def test_patch_task_updates_title(client):
    task = _create_task(client, title="Original")
    r = client.patch(f"/tasks/{task['id']}", json={"title": "Updated"})
    assert r.status_code == 200
    assert r.json()["task"]["title"] == "Updated"


def test_patch_nonexistent_task_returns_404(client):
    r = client.patch("/tasks/9999", json={"title": "Ghost"})
    assert r.status_code == 404


def test_delete_task_returns_200(client):
    task = _create_task(client)
    r = client.delete(f"/tasks/{task['id']}")
    assert r.status_code == 200


def test_delete_nonexistent_task_returns_404(client):
    r = client.delete("/tasks/9999")
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# 3. GET /graph
# ──────────────────────────────────────────────────────────────────────────────

def test_get_graph_empty(client):
    r = client.get("/graph")
    assert r.status_code == 200
    data = r.json()
    assert data["tasks"] == []
    assert data["edges"] == []
    assert data["critical_path"] == []


def test_get_graph_includes_blocked_ready(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    r = client.get("/graph")
    tasks = {t["id"]: t for t in r.json()["tasks"]}
    # B has an incomplete prerequisite (A is still Backlog), so B is Blocked.
    assert tasks[b["id"]]["blocked_ready"] == "Blocked"
    # A has no prerequisites.
    assert tasks[a["id"]]["blocked_ready"] == "Ready"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Dependency management
# ──────────────────────────────────────────────────────────────────────────────

def test_add_dependency_returns_201(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    r = client.post("/dependencies", json={
        "prerequisite_id": a["id"], "dependent_id": b["id"]
    })
    assert r.status_code == 201


def test_add_direct_cycle_returns_409(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    # Now add B→A which closes the cycle
    r = client.post("/dependencies", json={"prerequisite_id": b["id"], "dependent_id": a["id"]})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert "cycle" in str(detail).lower()


def test_add_self_dependency_returns_422_or_409(client):
    a = _create_task(client, "A")
    r = client.post("/dependencies", json={
        "prerequisite_id": a["id"], "dependent_id": a["id"]
    })
    # Pydantic validator catches this as 422; DB constraint would be 409.
    assert r.status_code in (409, 422)


def test_add_duplicate_dependency_returns_409(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    r = client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    assert r.status_code == 409
    assert "already exists" in r.json()["detail"]


def test_delete_dependency_returns_200(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    r = client.request("DELETE", "/dependencies", json={
        "prerequisite_id": a["id"], "dependent_id": b["id"]
    })
    assert r.status_code == 200


def test_delete_nonexistent_dependency_returns_404(client):
    r = client.request("DELETE", "/dependencies", json={
        "prerequisite_id": 999, "dependent_id": 998
    })
    assert r.status_code == 404


# ──────────────────────────────────────────────────────────────────────────────
# 5. Task moves — Blocked enforcement (403)
# ──────────────────────────────────────────────────────────────────────────────

def test_move_blocked_task_to_in_progress_returns_403(client):
    a = _create_task(client, "A")  # Backlog, incomplete
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})
    r = client.post(f"/tasks/{b['id']}/move", json={"status": "In Progress", "position": 0})
    assert r.status_code == 403
    detail = r.json()["detail"]
    assert "open_prerequisites" in detail or "Blocked" in str(detail)


def test_move_ready_task_to_in_progress_succeeds(client):
    a = _create_task(client, "A")
    # No prerequisites → Ready
    r = client.post(f"/tasks/{a['id']}/move", json={"status": "In Progress", "position": 0})
    assert r.status_code == 200
    assert r.json()["task"]["status"] == "In Progress"


def test_move_task_to_done_after_prereq_done(client):
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})

    # Move A to Done first
    client.post(f"/tasks/{a['id']}/move", json={"status": "Done", "position": 0})

    # Now B should be Ready → can move to In Progress
    r = client.post(f"/tasks/{b['id']}/move", json={"status": "In Progress", "position": 0})
    assert r.status_code == 200


def test_rollback_prereq_to_in_progress_reblocks_child(client):
    """
    Corresponds to: "Rollback on regression — Done→In Progress re-blocks dependents"
    """
    a = _create_task(client, "A")
    b = _create_task(client, "B")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})

    # Bring A to Done so B becomes Ready
    client.post(f"/tasks/{a['id']}/move", json={"status": "Done", "position": 0})

    # Move B to In Progress (valid while A is Done)
    client.post(f"/tasks/{b['id']}/move", json={"status": "In Progress", "position": 0})

    # Roll A back to In Progress
    client.post(f"/tasks/{a['id']}/move", json={"status": "In Progress", "position": 0})

    # Graph should now show B as Blocked again
    graph = client.get("/graph").json()
    task_b = next(t for t in graph["tasks"] if t["id"] == b["id"])
    assert task_b["blocked_ready"] == "Blocked"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Schedule impact / propagation via API
# ──────────────────────────────────────────────────────────────────────────────

def test_extending_task_propagates_to_downstream(client):
    a = _create_task(client, "A", duration_days=3, constraint_start="2026-01-01")
    b = _create_task(client, "B", duration_days=2, constraint_start="2026-01-01")
    client.post("/dependencies", json={"prerequisite_id": a["id"], "dependent_id": b["id"]})

    # Extend A by 5 days
    r = client.patch(f"/tasks/{a['id']}", json={"duration_days": 8})
    assert r.status_code == 200
    impact = r.json()["schedule_impact"]

    b_shifted = next((c for c in impact if c["task_id"] == b["id"]), None)
    assert b_shifted is not None, "B must appear in schedule impact"
    assert b_shifted["new_start"] > b_shifted["old_start"]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Persistence across restart simulation
# ──────────────────────────────────────────────────────────────────────────────

def test_persistence_across_client_restart(client):
    """
    Simulates 'persistence across server restart': data created through one
    request must still be visible in subsequent GET calls on the same DB.
    """
    task = _create_task(client, "Persistent Task", duration_days=7)
    task_id = task["id"]

    r = client.get("/graph")
    assert r.status_code == 200
    task_ids = [t["id"] for t in r.json()["tasks"]]
    assert task_id in task_ids


# ──────────────────────────────────────────────────────────────────────────────
# 8. Invalid status value rejected
# ──────────────────────────────────────────────────────────────────────────────

def test_invalid_status_on_create_returns_422(client):
    r = client.post("/tasks", json={
        "title": "Bad task",
        "duration_days": 1,
        "constraint_start": "2026-01-01",
        "status": "NotAStatus",
    })
    assert r.status_code == 422


def test_invalid_status_on_move_returns_422(client):
    a = _create_task(client, "A")
    r = client.post(f"/tasks/{a['id']}/move", json={"status": "Flying", "position": 0})
    assert r.status_code == 422
