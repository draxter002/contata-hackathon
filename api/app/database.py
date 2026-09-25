"""
database.py — SQLAlchemy engine and session factory.

SQLite is used for local development (zero setup, just run the server).
PostgreSQL is used in production via DATABASE_URL env var.

SQLite needs check_same_thread=False because FastAPI runs request handlers
in threadpools. PostgreSQL doesn't need that flag, so we apply it
conditionally.
"""

from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

from app.config import settings


def _build_engine():
    url = settings.effective_database_url
    connect_args = {}

    if settings.is_sqlite:
        # SQLite's default threading model rejects cross-thread use.
        # FastAPI's threadpool requires us to relax this.
        connect_args["check_same_thread"] = False

    engine = create_engine(
        url,
        connect_args=connect_args,
        # Keep a small pool; SQLite is single-writer anyway.
        pool_pre_ping=True,
    )

    if settings.is_sqlite:
        # Enable WAL mode and foreign key enforcement for SQLite.
        # WAL improves concurrent read performance marginally.
        # Foreign keys are OFF by default in SQLite — we must enable them
        # explicitly per connection.
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine = _build_engine()

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and ensures it is
    closed after the request completes — even on exception.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
