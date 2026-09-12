from fastapi import FastAPI

from infer.predict import router as infer_router

app = FastAPI(title="mood-ml", version="0.1.0")

app.include_router(infer_router, prefix="/internal/v1")
