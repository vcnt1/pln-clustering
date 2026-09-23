"""HTTP client for mood-ml's internal inference endpoint (data-model.md §2.2)."""

import os
from datetime import datetime
from typing import Optional

import httpx
from pydantic import BaseModel

MOOD_ML_BASE_URL = os.environ.get("MOOD_ML_BASE_URL", "http://localhost:8001")
INFER_TIMEOUT_SECONDS = float(os.environ.get("MOOD_ML_TIMEOUT_SECONDS", "2.0"))


class HistoryMessage(BaseModel):
    message_id: str
    role: str
    text: str


class InferRequest(BaseModel):
    request_id: str
    customer_id: str
    conversation_id: str
    history: list[HistoryMessage]


class InferResult(BaseModel):
    score: float
    scale: str
    mood_label: Optional[str] = None
    model_version: str
    computed_at: datetime


class InferenceError(Exception):
    """Raised for any inference outcome that must become an inference_failures row (P3)."""

    def __init__(self, error_code: str, detail: str):
        self.error_code = error_code
        self.detail = detail
        super().__init__(detail)


def call_infer(payload: InferRequest) -> InferResult:
    try:
        response = httpx.post(
            f"{MOOD_ML_BASE_URL}/internal/v1/infer",
            json=payload.model_dump(mode="json"),
            timeout=INFER_TIMEOUT_SECONDS,
        )
    except httpx.TimeoutException as exc:
        raise InferenceError("timeout", str(exc)) from exc
    except httpx.HTTPError as exc:
        raise InferenceError("ml_unavailable", str(exc)) from exc

    if response.status_code != 200:
        detail = _error_detail(response)
        error_code = "ml_unavailable" if response.status_code == 503 else "invalid_response"
        raise InferenceError(error_code, detail)

    try:
        data = response.json()
        return InferResult(
            score=data["score"],
            scale=data["scale"],
            mood_label=data.get("mood_label"),
            model_version=data["model_version"],
            computed_at=data["computed_at"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise InferenceError("invalid_response", str(exc)) from exc


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail", response.text))
    except ValueError:
        return response.text
