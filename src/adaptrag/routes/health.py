from fastapi import APIRouter

from adaptrag import __version__

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
