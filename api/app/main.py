"""
main.py — FastAPI application entry point for TaskFlow Pro.

Security decisions encoded here:
- CORS uses an explicit allow-list, not a wildcard. In dev, localhost:3000
  is always permitted. In production, ALLOWED_ORIGIN from env is added.
- The API key is never logged or returned to clients. It stays in the
  settings object accessed only by ai_service.py.
- All tables are created on startup via Base.metadata.create_all —
  zero-setup for local SQLite. In production, run Alembic migrations instead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
import app.database as _db_module
from app.models import Base
from app.api.tasks import router as tasks_router, graph_router
from app.api.dependencies import router as deps_router
from app.api.ai import router as ai_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use the module-level engine so tests can patch app.database.engine.
    Base.metadata.create_all(bind=_db_module.engine)
    logger.info(
        "TaskFlow Pro API started. DB: %s",
        "SQLite (local)" if settings.is_sqlite else "PostgreSQL",
    )
    yield
    logger.info("TaskFlow Pro API shutting down.")



app = FastAPI(
    title="TaskFlow Pro API",
    version="1.0.0",
    description=(
        "DAG-aware project task manager with schedule propagation "
        "and AI-assisted dependency suggestions."
    ),
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# We always allow localhost:3000 for local development.
# In production, add ALLOWED_ORIGIN from env.
_allowed_origins = ["http://localhost:3000"]
if settings.allowed_origin and settings.allowed_origin not in _allowed_origins:
    _allowed_origins.append(settings.allowed_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Accept"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(graph_router)
app.include_router(tasks_router)
app.include_router(deps_router)
app.include_router(ai_router)


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "db": settings.effective_database_url.split("://")[0]}
