from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.database.connection import close_db_pool, init_db_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Initialize DB connection pool
    init_db_pool()
    print("✓ Database connection pool initialized")

    yield

    # Shutdown: Close DB connection pool
    close_db_pool()
    print("✓ Database connection pool closed")


# Create FastAPI app
app = FastAPI(
    title="Retrieval Evals API",
    description="YouTube transcript retrieval system with Postgres FTS + Semantic re-ranking",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware (for UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api")


# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "retrieval-evals",
        "milestone": "M1",
        "endpoints": [
            "POST /api/ingest",
            "POST /api/query",
            "GET /api/bench/retrieval",
            "GET /api/health",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
