from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    conversation_id: str
    customer_id: str
    role: Literal["customer", "agent"]
    message: str
    timestamp: datetime


class IngestResponse(BaseModel):
    status: str
    message_id: str


class MoodResponse(BaseModel):
    customer_id: str
    conversation_id: str
    score: float
    scale: str
    mood_label: Optional[str] = None
    model_version: str
    computed_at: datetime


class ConversationMessage(BaseModel):
    message_id: str
    role: Literal["customer", "agent"]
    text: str
    sent_at: datetime


class ConversationSummary(BaseModel):
    conversation_id: str
    customer_id: str
    last_message_at: datetime
    score: Optional[float] = None
    scale: Optional[str] = None
    messages: list[ConversationMessage]
