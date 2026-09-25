"""
graph_service.py — Shared helper that converts DB rows into the engine's
in-memory format and back, and coordinates all graph mutations.

This module is the bridge between the persistence layer (SQLAlchemy) and
the pure computation layer (dag_engine.py). It deliberately owns no HTTP
concerns — it receives a Session and returns plain Python objects that the
routers then serialise into responses.

Why a separate service layer?
  We want the routers to be thin (validate input → call service → return
  response) and the engine to be framework-free. This service is where the
  two worlds meet. It also means we can test the service independently of
  both HTTP and the pure engine.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.models import Dependency, DependencySuggestion, SuggestionState, Task
from app.engine.dag_engine import (
    check_cycle_with_path,
    compute_critical_path,
    compute_initial_dates,
    derive_blocked_ready,
    get_open_prerequisites,
    is_move_valid_for_blocked,
    recalculate_schedule,
    topological_sort,
)


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot helpers — convert DB rows to engine-compatible dicts
# ──────────────────────────────────────────────────────────────────────────────

def _task_to_node(task: Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "status": task.status,
        "duration_days": task.duration_days,
        "constraint_start": task.constraint_start,
        "start_date": task.start_date,
        "end_date": task.end_date,
    }


def _load_graph_snapshot(
    db: Session,
) -> tuple[dict[int, dict], dict[int, set[int]]]:
    """
    Load all tasks and edges from the DB into the engine's in-memory format.
    Returns (tasks_map, adjacency_list).
    """
    tasks_map: dict[int, dict] = {
        t.id: _task_to_node(t) for t in db.query(Task).all()
    }
    adj: dict[int, set[int]] = {tid: set() for tid in tasks_map}
    for dep in db.query(Dependency).all():
        adj.setdefault(dep.prerequisite_id, set()).add(dep.dependent_id)
        adj.setdefault(dep.dependent_id, set())
    return tasks_map, adj


def _reverse_adj(adj: dict[int, set[int]]) -> dict[int, set[int]]:
    rev: dict[int, set[int]] = {tid: set() for tid in adj}
    for prereq, deps in adj.items():
        for dep in deps:
            rev.setdefault(dep, set()).add(prereq)
    return rev


def _enrich_task(
    task: Task,
    prereq_ids: set[int],
    tasks_map: dict[int, dict],
) -> dict[str, Any]:
    """Add the derived blocked_ready field to a task dict for serialisation."""
    node = _task_to_node(task)
    node["blocked_ready"] = derive_blocked_ready(node, prereq_ids, tasks_map)
    node["created_at"] = task.created_at
    node["updated_at"] = task.updated_at
    node["description"] = task.description or ""
    node["position"] = task.position
    return node


# ──────────────────────────────────────────────────────────────────────────────
# Graph read
# ──────────────────────────────────────────────────────────────────────────────

def get_full_graph(db: Session) -> dict[str, Any]:
    """
    Build and return the full graph payload for GET /graph.
    Computes blocked_ready and critical_path on the fly.
    """
    tasks_map, adj = _load_graph_snapshot(db)
    rev = _reverse_adj(adj)

    all_tasks_db = {t.id: t for t in db.query(Task).all()}
    enriched_tasks = [
        _enrich_task(all_tasks_db[tid], rev.get(tid, set()), tasks_map)
        for tid in tasks_map
    ]

    edges = [
        {"prerequisite_id": dep.prerequisite_id, "dependent_id": dep.dependent_id}
        for dep in db.query(Dependency).all()
    ]

    critical_path = compute_critical_path(tasks_map, adj)

    return {
        "tasks": enriched_tasks,
        "edges": edges,
        "critical_path": critical_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task CRUD
# ──────────────────────────────────────────────────────────────────────────────

def create_task(db: Session, data: dict[str, Any]) -> tuple[Task, list]:
    """
    Create a new task. start_date defaults to constraint_start since a new
    task has no prerequisites yet. Returns (task, []) — no schedule impact
    on creation (only this task is new, no existing tasks shift).
    """
    cs = data["constraint_start"]
    dur = data.get("duration_days", 1)

    task = Task(
        title=data["title"],
        description=data.get("description", ""),
        status=data.get("status", "Backlog"),
        position=data.get("position", 0),
        duration_days=dur,
        constraint_start=cs,
        start_date=cs,
        end_date=cs + timedelta(days=dur),
    )
    db.add(task)
    db.flush()  # get the id without committing yet
    return task, []


def update_task(db: Session, task: Task, updates: dict[str, Any]) -> list:
    """
    Apply field updates to a task and re-propagate schedules for its
    descendants if duration or constraint_start changed.
    Returns a list of ScheduleChangeRecords.
    """
    date_affecting_fields = {"duration_days", "constraint_start"}
    dates_changed = any(k in updates for k in date_affecting_fields)

    for field, value in updates.items():
        setattr(task, field, value)

    db.flush()

    if not dates_changed:
        return []

    tasks_map, adj = _load_graph_snapshot(db)
    # Re-derive the task's own end_date with the new values before propagating.
    node = tasks_map[task.id]
    node["duration_days"] = task.duration_days
    node["constraint_start"] = task.constraint_start
    node["start_date"] = task.start_date
    node["end_date"] = task.start_date + timedelta(days=task.duration_days)

    changes = recalculate_schedule(tasks_map, adj, changed_task_id=task.id)
    _write_schedule_changes(db, changes, tasks_map)
    return changes


def delete_task(db: Session, task: Task) -> list:
    """
    Delete a task. Cascade handles edge removal. We then re-propagate the
    remaining graph so surviving tasks get correct dates.
    Returns schedule impact for the remaining tasks.
    """
    task_id = task.id
    db.delete(task)
    db.flush()

    # After deletion, reload and re-propagate from scratch for the
    # remaining graph. We use compute_initial_dates since we don't know
    # which specific subtree was affected.
    tasks_map, adj = _load_graph_snapshot(db)
    if not tasks_map:
        return []

    changes = compute_initial_dates(tasks_map, adj)
    _write_schedule_changes(db, changes, tasks_map)
    return changes


def move_task(
    db: Session, task: Task, target_status: str, target_position: int
) -> tuple[Task, list]:
    """
    Move a task to a new status/position column.
    Returns 403 data if the move is rejected (Blocked task moving forward).
    Raises ValueError with open_prerequisites list on rejection.
    """
    tasks_map, adj = _load_graph_snapshot(db)
    rev = _reverse_adj(adj)
    prereq_ids = rev.get(task.id, set())
    blocked_ready = derive_blocked_ready(tasks_map[task.id], prereq_ids, tasks_map)

    if not is_move_valid_for_blocked(blocked_ready, target_status):
        open_prereqs = get_open_prerequisites(tasks_map[task.id], prereq_ids, tasks_map)
        raise PermissionError(open_prereqs)

    task.status = target_status
    task.position = target_position
    db.flush()
    return task, []


# ──────────────────────────────────────────────────────────────────────────────
# Dependency CRUD
# ──────────────────────────────────────────────────────────────────────────────

def add_dependency(
    db: Session, prerequisite_id: int, dependent_id: int
) -> tuple[Dependency, list]:
    """
    Validate and add a dependency edge.
    Raises ValueError with cycle path if a cycle would result.
    Raises LookupError if either task doesn't exist.
    Raises KeyError on duplicate edge.
    Returns (dependency, schedule_impact).

    Note: we check for duplicates BEFORE the task existence check so that
    the 409 response fires correctly even when session state is complex.
    """
    # Check for existing duplicate edge first (returns 409 via KeyError).
    existing = (
        db.query(Dependency)
        .filter_by(prerequisite_id=prerequisite_id, dependent_id=dependent_id)
        .first()
    )
    if existing:
        raise KeyError("This dependency already exists")

    # Both tasks must exist.
    prereq = db.get(Task, prerequisite_id)
    dep_task = db.get(Task, dependent_id)
    if prereq is None or dep_task is None:
        raise LookupError("One or both task IDs not found")

    tasks_map, adj = _load_graph_snapshot(db)

    # Cycle check before writing anything.
    cycle_path = check_cycle_with_path(adj, prerequisite_id, dependent_id)
    if cycle_path is not None:
        raise ValueError(cycle_path)

    dep = Dependency(prerequisite_id=prerequisite_id, dependent_id=dependent_id)
    db.add(dep)
    db.flush()

    # After adding the edge, re-propagate from the prerequisite task.
    tasks_map, adj = _load_graph_snapshot(db)
    changes = recalculate_schedule(tasks_map, adj, changed_task_id=prerequisite_id)
    _write_schedule_changes(db, changes, tasks_map)
    return dep, changes



def remove_dependency(db: Session, prerequisite_id: int, dependent_id: int) -> list:
    """
    Remove a dependency edge and re-propagate the graph.
    The dependent task may move earlier now that the constraint is gone.
    """
    dep = (
        db.query(Dependency)
        .filter_by(prerequisite_id=prerequisite_id, dependent_id=dependent_id)
        .first()
    )
    if dep is None:
        raise LookupError("Dependency not found")

    db.delete(dep)
    db.flush()

    tasks_map, adj = _load_graph_snapshot(db)
    changes = compute_initial_dates(tasks_map, adj)
    _write_schedule_changes(db, changes, tasks_map)
    return changes


# ──────────────────────────────────────────────────────────────────────────────
# Schedule write-back helper
# ──────────────────────────────────────────────────────────────────────────────

def _write_schedule_changes(
    db: Session,
    changes: list[dict],
    tasks_map: dict[int, dict],
) -> None:
    """
    Persist the dates computed by the engine back to the DB rows.
    We fetch tasks by ID and update in bulk rather than one-by-one to
    avoid N+1 queries on long chains.
    """
    if not changes:
        return
    changed_ids = {c["task_id"] for c in changes}
    tasks_db = {
        t.id: t for t in db.query(Task).filter(Task.id.in_(changed_ids)).all()
    }
    for change in changes:
        task = tasks_db.get(change["task_id"])
        if task:
            task.start_date = tasks_map[change["task_id"]]["start_date"]
            task.end_date = tasks_map[change["task_id"]]["end_date"]
    db.flush()
