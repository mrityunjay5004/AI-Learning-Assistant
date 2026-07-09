"""
FastAPI application entrypoint.

Wires together: logging config, DB initialization, exception handlers,
request-latency logging middleware, and the feature routers (roadmap,
project, chat).
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.db.database import init_db
from app.exceptions import register_exception_handlers
from app.logging_config import configure_logging
from app.routers import chat, project, roadmap

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Learning Assistant API...")
    init_db()
    logger.info("Database initialized.")
    yield
    logger.info("Shutting down AI Learning Assistant API.")


app = FastAPI(
    title="AI Learning Assistant",
    description=(
        "Backend service that generates personalized learning roadmaps, "
        "recommends projects, and answers roadmap-grounded questions via RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

register_exception_handlers(app)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request's method, path, status code, and wall-clock
    latency - independent of the more detailed LLM/retrieval timing logged
    inside individual services, this gives an end-to-end view per request
    without needing an external APM tool."""
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000
    logger.info(
        "%s %s -> %d (%.0fms)",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.get("/health", tags=["Health"])
def health_check() -> dict:
    return {"status": "ok"}


app.include_router(roadmap.router)
app.include_router(project.router)
app.include_router(chat.router)
