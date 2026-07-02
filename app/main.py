"""FastAPI application entrypoint."""

import logging

from fastapi import FastAPI

from app.api.routes_documents import router as documents_router
from app.api.routes_health import router as health_router
from app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    configure_logging()
    logger.info("Starting Local Document Q&A API")
    app = FastAPI(
        title="Local Document Q&A API",
        version="0.1.0",
        description=(
            "Local document analysis and question-answering API for the "
            "TUSAS technical assessment. Day 1 includes health checks, "
            "Ollama connectivity checks, and upload infrastructure only."
        ),
        contact={"name": "TUSAS Technical Assessment"},
        license_info={"name": "Internal evaluation"},
    )
    app.include_router(health_router)
    app.include_router(documents_router)
    return app


app = create_app()


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    """Return a basic API running message."""
    return {"status": "ok", "message": "Local Document Q&A API is running."}
