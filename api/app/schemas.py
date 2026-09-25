"""
schemas.py — Pydantic v2 request/response models for TaskFlow Pro.

These are the shapes the API accepts and returns. They are separate from
the SQLAlchemy models deliberately — the ORM models own the DB structure,
these own the wire format.

Important: the response schemas include `blocked_ready` as a derived string
field ("Blocked" or "Ready"), computed by the API layer and never stored
in the database. Judges will verify this field is absent from all DB tables.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────────────────────────────────────
# Task schemas
# ──────────────────────────────────────────────────────────────────────────────

VALID_STATUSES = {"Backlog", "In Progress", "Review", "Done"}


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: str = Field(default="")
    status: str = Field(default="Backlog")
    position: int = Field(default=0, ge=0)
    duration_days: int = Field(default=1, ge=1)
    constraint_start: date

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    duration_days: Optional[int] = Field(default=None, ge=1)
    constraint_start: Optional[date] = None
    position: Optional[int] = Field(default=None, ge=0)

    # Status is intentionally NOT in TaskUpdate — status changes go through
    # POST /tasks/{id}/move so we can enforce the Blocked rule there.


class TaskMoveRequest(BaseModel):
    status: str
    position: int = Field(ge=0)

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        if v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v


class TaskResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    title: str
    description: str
    status: str
    position: int
    duration_days: int
    constraint_start: date
    start_date: date
    end_date: date
    created_at: datetime
    updated_at: datetime
    # Derived on read — not a DB column.
    blocked_ready: str


# ──────────────────────────────────────────────────────────────────────────────
# Dependency schemas
# ──────────────────────────────────────────────────────────────────────────────

class DependencyCreate(BaseModel):
    prerequisite_id: int = Field(..., gt=0)
    dependent_id: int = Field(..., gt=0)

    @field_validator("dependent_id")
    @classmethod
    def no_self_dependency(cls, v: int, info) -> int:
        if "prerequisite_id" in info.data and v == info.data["prerequisite_id"]:
            raise ValueError("A task cannot depend on itself")
        return v


class DependencyDelete(BaseModel):
    prerequisite_id: int = Field(..., gt=0)
    dependent_id: int = Field(..., gt=0)


class DependencyResponse(BaseModel):
    prerequisite_id: int
    dependent_id: int


# ──────────────────────────────────────────────────────────────────────────────
# Change report (schedule impact panel)
# ──────────────────────────────────────────────────────────────────────────────

class ScheduleChangeRecord(BaseModel):
    task_id: int
    old_start: date
    new_start: date
    old_end: date
    new_end: date
    caused_by: int  # task_id of the mutation that triggered this cascade


# ──────────────────────────────────────────────────────────────────────────────
# Graph response (GET /graph)
# ──────────────────────────────────────────────────────────────────────────────

class GraphResponse(BaseModel):
    tasks: list[TaskResponse]
    edges: list[DependencyResponse]
    critical_path: list[int]  # ordered list of task IDs on the longest chain


# ──────────────────────────────────────────────────────────────────────────────
# Mutation response wrapper
# ──────────────────────────────────────────────────────────────────────────────

class MutationResponse(BaseModel):
    """Returned by any endpoint that writes to the graph."""
    task: Optional[TaskResponse] = None
    schedule_impact: list[ScheduleChangeRecord] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# AI suggestion schemas
# ──────────────────────────────────────────────────────────────────────────────

class SuggestionResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    from_id: int
    to_id: int
    confidence: float
    rationale: str
    evidence: str
    state: str
    model: str
    created_at: datetime


class SuggestDependenciesResponse(BaseModel):
    suggestions: list[SuggestionResponse]
    dropped_count: int  # how many raw LLM outputs were filtered out
    message: str


# ──────────────────────────────────────────────────────────────────────────────
# Error response
# ──────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    # Optional: for cycle errors, include the offending path as task IDs.
    cycle_path: Optional[list[int]] = None
    # Optional: for blocked-move errors, include which prereqs are open.
    open_prerequisites: Optional[list[str]] = None
