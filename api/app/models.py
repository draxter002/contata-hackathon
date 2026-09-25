"""
models.py — SQLAlchemy ORM models for TaskFlow Pro.

Table names and column names match ARCHITECTURE.md exactly.

Key design notes:
- Blocked/Ready is NOT a column. It is derived on read in the API layer
  by calling derive_blocked_ready() from dag_engine.py.
- start_date and end_date are stored (not recomputed on every read), but
  always written by the DAG engine after any mutation — never by raw input.
- We use SQLAlchemy 2.0 Mapped[] annotations throughout to satisfy the
  new DeclarativeBase type-checking rules.
"""

from __future__ import annotations

import enum
from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class TaskStatus(str, enum.Enum):
    BACKLOG = "Backlog"
    IN_PROGRESS = "In Progress"
    REVIEW = "Review"
    DONE = "Done"


class SuggestionState(str, enum.Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default="")
    status: Mapped[str] = mapped_column(
        Enum(TaskStatus, name="task_status_enum", native_enum=False),
        nullable=False,
        default=TaskStatus.BACKLOG,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_days: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    constraint_start: Mapped[date] = mapped_column(Date, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # A Task can appear as the prerequisite side of many Dependency rows.
    as_prerequisite: Mapped[List["Dependency"]] = relationship(
        "Dependency",
        foreign_keys="Dependency.prerequisite_id",
        back_populates="prerequisite_task",
        cascade="all, delete-orphan",
    )
    # A Task can appear as the dependent side of many Dependency rows.
    as_dependent: Mapped[List["Dependency"]] = relationship(
        "Dependency",
        foreign_keys="Dependency.dependent_id",
        back_populates="dependent_task",
        cascade="all, delete-orphan",
    )


class Dependency(Base):
    __tablename__ = "dependencies"

    prerequisite_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    dependent_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("tasks.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("prerequisite_id", "dependent_id", name="uq_dependency_pair"),
        CheckConstraint(
            "prerequisite_id != dependent_id", name="ck_no_self_dependency"
        ),
    )

    prerequisite_task: Mapped["Task"] = relationship(
        "Task", foreign_keys=[prerequisite_id], back_populates="as_prerequisite"
    )
    dependent_task: Mapped["Task"] = relationship(
        "Task", foreign_keys=[dependent_id], back_populates="as_dependent"
    )


class DependencySuggestion(Base):
    __tablename__ = "dependency_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    to_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(
        Enum(SuggestionState, name="suggestion_state_enum", native_enum=False),
        nullable=False,
        default=SuggestionState.PENDING,
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    from_task: Mapped["Task"] = relationship("Task", foreign_keys=[from_id])
    to_task: Mapped["Task"] = relationship("Task", foreign_keys=[to_id])
