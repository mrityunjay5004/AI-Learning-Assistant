"""
Custom application exceptions and their FastAPI exception handlers.

Keeping these centralized means every router raises meaningful, typed
errors (RoadmapNotFoundError, LLMGenerationError, ...) and the client always
gets a consistent JSON error shape instead of a raw 500 traceback.
"""
import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all application-level errors."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    default_message = "Internal server error."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class RoadmapNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    default_message = "Roadmap not found."


class InvalidRequestError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    default_message = "Invalid request."


class LLMGenerationError(AppError):
    """Raised when the LLM fails to produce a usable, valid response after retries."""

    status_code = status.HTTP_502_BAD_GATEWAY
    default_message = "The AI model failed to generate a valid response. Please try again."


class LLMTimeoutError(AppError):
    """Raised when the LLM does not respond within an acceptable time budget."""

    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    default_message = "The AI model took too long to respond. Please try again."


class EmbeddingError(AppError):
    """Raised when generating an embedding (for indexing or querying) fails."""

    status_code = status.HTTP_502_BAD_GATEWAY
    default_message = "Failed to generate embeddings for retrieval."


class VectorStoreError(AppError):
    """Raised when the vector store (Chroma) fails to read or write."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "The retrieval store is temporarily unavailable."


class DatabaseError(AppError):
    """Raised when a database read/write fails unexpectedly."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    default_message = "A database error occurred. Please try again."


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning("AppError on %s %s: %s", request.method, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.__class__.__name__, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )
