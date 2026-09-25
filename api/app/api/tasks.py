"""
tasks.py — FastAPI router for task CRUD and moves.

Every mutating operation runs inside a single DB transaction (the session
is committed once at the end of the request, or rolled back on exception).
The transaction boundary is managed by the router, not the service layer,
so the service stays testable without HTTP.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Task
from app.schemas import (
    ErrorResponse,
    GraphResponse,
    MutationResponse,
    ScheduleChangeRecord,
    TaskCreate,
    TaskMoveRequest,
    TaskResponse,
    TaskUpdate,
)
from app.services import graph_service

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _task_to_response(task_dict: dict) -> TaskResponse:
    return TaskResponse(**task_dict)


# ──────────────────────────────────────────────────────────────────────────────
# GET /graph — full graph with derived Blocked/Ready and critical path
# ──────────────────────────────────────────────────────────────────────────────

graph_router = APIRouter(tags=["graph"])


@graph_router.get("/graph", response_model=GraphResponse)
def get_graph(db: Session = Depends(get_db)):
    data = graph_service.get_full_graph(db)
    return GraphResponse(
        tasks=[TaskResponse(**t) for t in data["tasks"]],
        edges=data["edges"],
        critical_path=data["critical_path"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# POST /tasks — create a task
# ──────────────────────────────────────────────────────────────────────────────

@router.post("", response_model=MutationResponse, status_code=201)
def create_task(body: TaskCreate, db: Session = Depends(get_db)):
    try:
        task, changes = graph_service.create_task(db, body.model_dump())
        db.commit()
        db.refresh(task)
        task_data = graph_service._enrich_task(
            task, set(), {task.id: graph_service._task_to_node(task)}
        )
        return MutationResponse(
            task=TaskResponse(**task_data),
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# PATCH /tasks/{id} — update task fields
# ──────────────────────────────────────────────────────────────────────────────

@router.patch("/{task_id}", response_model=MutationResponse)
def update_task(task_id: int, body: TaskUpdate, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")

    try:
        changes = graph_service.update_task(db, task, updates)
        db.commit()
        db.refresh(task)
        # Reload full graph for enriched response
        full = graph_service.get_full_graph(db)
        task_data = next((t for t in full["tasks"] if t["id"] == task_id), None)
        return MutationResponse(
            task=TaskResponse(**task_data) if task_data else None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# DELETE /tasks/{id} — delete a task (after frontend confirmation)
# ──────────────────────────────────────────────────────────────────────────────

@router.delete("/{task_id}", response_model=MutationResponse)
def delete_task(task_id: int, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        changes = graph_service.delete_task(db, task)
        db.commit()
        return MutationResponse(
            task=None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# POST /tasks/{id}/move — column/status change with Blocked enforcement
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/{task_id}/move", response_model=MutationResponse)
def move_task(task_id: int, body: TaskMoveRequest, db: Session = Depends(get_db)):
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    try:
        updated_task, changes = graph_service.move_task(
            db, task, body.status, body.position
        )
        db.commit()
        db.refresh(updated_task)
        full = graph_service.get_full_graph(db)
        task_data = next((t for t in full["tasks"] if t["id"] == task_id), None)
        return MutationResponse(
            task=TaskResponse(**task_data) if task_data else None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except PermissionError as open_prereqs:
        db.rollback()
        prereq_list = list(open_prereqs.args[0]) if open_prereqs.args else []
        raise HTTPException(
            status_code=403,
            detail={
                "message": (
                    f"Cannot move a Blocked task to '{body.status}'. "
                    f"These prerequisites are not yet Done: {prereq_list}"
                ),
                "open_prerequisites": prereq_list,
            },
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
