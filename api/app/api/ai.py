"""
ai.py — FastAPI router for AI dependency suggestions.

Rate limiting is handled with a simple in-memory counter per the spec.
The /suggest-dependencies endpoint is the ONLY place the LLM is called —
nothing automatic, nothing on page load.

503 is returned (not 500) when the AI service is unavailable, so the
client can distinguish "server error" from "feature unavailable".
"""

from __future__ import annotations

import time
from collections import deque

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import DependencySuggestion, SuggestionState
from app.schemas import (
    MutationResponse,
    ScheduleChangeRecord,
    SuggestDependenciesResponse,
    SuggestionResponse,
)
from app.services import ai_service, graph_service

router = APIRouter(prefix="/ai", tags=["ai"])

# ── Simple in-memory rate limiter ─────────────────────────────────────────────
# Allow at most 5 calls per 60 seconds. This is per-process, not per-user
# (we have no auth), which is fine for single-workspace scope.
_RATE_LIMIT_MAX = 5
_RATE_LIMIT_WINDOW = 60  # seconds
_request_timestamps: deque[float] = deque()


def _check_rate_limit() -> None:
    now = time.time()
    # Drop timestamps older than the window.
    while _request_timestamps and now - _request_timestamps[0] > _RATE_LIMIT_WINDOW:
        _request_timestamps.popleft()
    if len(_request_timestamps) >= _RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail=(
                f"AI suggestion rate limit exceeded. "
                f"Max {_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s."
            ),
        )
    _request_timestamps.append(now)


# ──────────────────────────────────────────────────────────────────────────────
# POST /ai/suggest-dependencies
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/suggest-dependencies", response_model=SuggestDependenciesResponse)
def suggest_dependencies(db: Session = Depends(get_db)):
    """
    Trigger the LLM call that generates dependency suggestions.
    This is the only place in the codebase that calls the Anthropic API.
    The user must explicitly click the "Suggest dependencies" button
    to hit this endpoint — nothing calls it automatically.
    """
    _check_rate_limit()
    try:
        result = ai_service.run_ai_suggestion(db)
        db.commit()
        return SuggestDependenciesResponse(
            suggestions=[
                SuggestionResponse.model_validate(s) for s in result["suggestions"]
            ],
            dropped_count=result["dropped_count"],
            message=result["message"],
        )
    except RuntimeError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# POST /ai/suggestions/{id}/accept
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/suggestions/{suggestion_id}/accept", response_model=MutationResponse)
def accept_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    """
    Accept a suggestion by running it through the SAME validated add-dependency
    path used for manual edges (cycle check included). This is a deliberate
    design choice: accepting an AI suggestion is not a shortcut around
    validation — it goes through the exact same code path.
    """
    suggestion = db.get(DependencySuggestion, suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    if suggestion.state != SuggestionState.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Suggestion is already {suggestion.state}",
        )

    try:
        dep, changes = graph_service.add_dependency(
            db, suggestion.from_id, suggestion.to_id
        )
        suggestion.state = SuggestionState.ACCEPTED
        db.commit()
        return MutationResponse(
            task=None,
            schedule_impact=[ScheduleChangeRecord(**c) for c in changes],
        )
    except LookupError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc))
    except KeyError:
        # Edge already exists — mark accepted anyway since the intent is satisfied.
        suggestion.state = SuggestionState.ACCEPTED
        db.commit()
        return MutationResponse(task=None, schedule_impact=[])
    except ValueError as cycle_path:
        db.rollback()
        path = cycle_path.args[0] if cycle_path.args else []
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Accepting this suggestion would create a cycle.",
                "cycle_path": path,
            },
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# POST /ai/suggestions/{id}/reject
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/suggestions/{suggestion_id}/reject", response_model=SuggestionResponse)
def reject_suggestion(suggestion_id: int, db: Session = Depends(get_db)):
    """
    Reject a suggestion so it isn't re-suggested in future runs.
    The rejection is stored; the AI service skips suggestions that already
    appear in the suggestions table as rejected.
    """
    suggestion = db.get(DependencySuggestion, suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Suggestion not found")
    if suggestion.state != SuggestionState.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"Suggestion is already {suggestion.state}",
        )

    try:
        suggestion.state = SuggestionState.REJECTED
        db.commit()
        return SuggestionResponse.model_validate(suggestion)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc))
