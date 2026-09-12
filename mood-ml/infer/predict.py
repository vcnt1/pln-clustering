"""Computes the real-time mood temperature and exposes it to mood-api."""

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class InferRequest(BaseModel):
    customer_id: str
    conversation_id: str
    message: str


class InferResponse(BaseModel):
    customer_id: str
    score: float
    scale: str
    model_version: str
    computed_at: datetime


@router.post("/infer", response_model=InferResponse)
def infer_mood(payload: InferRequest) -> InferResponse:
    # TODO: replace placeholder score with the trained model's prediction.
    return InferResponse(
        customer_id=payload.customer_id,
        score=0.0,
        scale="neutral",
        model_version="untrained",
        computed_at=datetime.now(timezone.utc),
    )
