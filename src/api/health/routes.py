from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("")
async def health_check():
    """
    Health check endpoint.

    Returns service status and current milestone.
    """
    return {"status": "healthy"}
