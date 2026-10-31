from fastapi import FastAPI

from adaptrag import __version__
from adaptrag.config import get_settings
from adaptrag.routes import answer, documents, jobs, query, ready
from adaptrag.routes.health import router as health_router


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)

    app.include_router(health_router)
    app.include_router(documents.router)
    app.include_router(ready.router)
    app.include_router(jobs.router)
    app.include_router(query.router)
    app.include_router(answer.router)

    return app


app = create_app()
