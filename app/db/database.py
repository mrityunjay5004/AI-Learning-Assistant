"""
SQLAlchemy engine/session setup.

SQLite is used for simplicity (zero external services to run for this
assignment) but the code only touches the ORM layer, so swapping
DATABASE_URL to Postgres/MySQL in production requires no code changes.
"""
import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# Ensure the sqlite file's parent directory exists
db_path = settings.database_url.replace("sqlite:///", "")
if db_path and db_path != ":memory:":
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Session:
    """Context manager for use outside request handlers (e.g. scripts, retries)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create tables on startup. For this assignment's scope, no migration
    tool (Alembic) is used; a fresh create_all is sufficient."""
    from app.models import db_models  # noqa: F401  (register models with Base)

    Base.metadata.create_all(bind=engine)
