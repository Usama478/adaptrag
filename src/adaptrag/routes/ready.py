import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from adaptrag.db import get_session

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/ready")
def ready(session: Annotated[Session, Depends(get_session)]) -> dict[str, str]:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError as exc:
        logger.warning("Readiness check failed: %s", exc.orig)
        raise HTTPException(status_code=503, detail="Database is not reachable.") from exc
    return {"status": "ready", "database": "ok"}
