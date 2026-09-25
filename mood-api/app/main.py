import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api.routes import router
from app.db.connection import init_db

app = FastAPI(title="mood-api", version="0.1.0")

# Allows chat-app's dev server (a different origin) to call this API from the browser.
_allowed_origins = os.environ.get("MOOD_API_ALLOWED_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

app.include_router(router, prefix="/v1alpha1")

_OPENAPI_YAML_PATH = Path(__file__).resolve().parent.parent / "openapi.yaml"


@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml() -> FileResponse:
    return FileResponse(_OPENAPI_YAML_PATH, media_type="application/yaml")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "invalid request body"})


@app.on_event("startup")
def on_startup() -> None:
    init_db()
