from fastapi import FastAPI

from app.api.routes import router
from app.db.connection import init_db

app = FastAPI(title="mood-api", version="0.1.0")

app.include_router(router, prefix="/v1alpha1")


@app.on_event("startup")
def on_startup() -> None:
    init_db()
