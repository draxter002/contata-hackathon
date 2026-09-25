"""
dependencies.py — FastAPI router for dependency edge management.

409 Conflict is returned for both cycle attempts and duplicate edges.
Each response includes a human-readable message per spec.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import (
    DependencyCreate,
    DependencyDelete,
    DependencyResponse,
    MutationResponse,
    ScheduleChangeRecord,
)
from app.services import graph_service

router = APIRouter(prefix="/dependencies", tags=["dependencies"])


@router.post("", response_model=MutationResponse, status_code=201)
def add_dependency(body: DependencyCreate, db: Session = Depends(get_db)):
    try:
        dep, changes = graph_service.add_dependency(
            db, body.prerequisite_id, body.dependent_id
        )
        db.commit()
        return MutationResponse(
            task=None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except KeyError:
        # Duplicate edge — must come before LookupError since KeyError is a subclass.
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="This dependency already exists between these two tasks.",
        )
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as cycle_path:
        db.rollback()
        path = cycle_path.args[0] if cycle_path.args else []
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"Adding this dependency would create a cycle: "
                    f"{' → '.join(str(tid) for tid in path)}"
                ),
                "cycle_path": path,
            },
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))



@router.delete("", response_model=MutationResponse)
def remove_dependency(body: DependencyDelete, db: Session = Depends(get_db)):
    try:
        changes = graph_service.remove_dependency(
            db, body.prerequisite_id, body.dependent_id
        )
        db.commit()
        return MutationResponse(
            task=None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
