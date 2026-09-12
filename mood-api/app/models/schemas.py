from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class IngestRequest(BaseModel):
    conversation_id: str
    customer_id: str
    role: Literal["customer", "agent"]
    message: str
    timestamp: datetime


class MoodResponse(BaseModel):
    customer_id: str
    score: float
    scale: str
    model_version: str
    computed_at: datetime
