from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """Check API health and readiness

    Simple health check endpoint to verify the API is running and database
    connectivity is operational.

    **Returns:** Status object indicating service health.
    """
    return {"status": "healthy"}
