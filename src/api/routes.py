from fastapi import APIRouter

from src.api.chat import router as chat_router
from src.api.health import router as health_router
from src.api.ingest import router as ingest_router
from src.api.retrieval import router as retrieval_router

# Main API router
router = APIRouter()

# Include all sub-routers
router.include_router(chat_router)  # type: ignore[arg-type]
router.include_router(health_router)  # type: ignore[arg-type]
router.include_router(ingest_router)  # type: ignore[arg-type]
router.include_router(retrieval_router)  # type: ignore[arg-type]
