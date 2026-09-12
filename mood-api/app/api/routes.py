import uuid

from fastapi import APIRouter, HTTPException

from app.db.connection import get_connection
from app.models.schemas import IngestRequest, MoodResponse

router = APIRouter()


@router.post("/ingest", status_code=200)
def ingest_message(payload: IngestRequest) -> dict:
    con = get_connection()
    con.execute(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?)",
        [
            str(uuid.uuid4()),
            payload.conversation_id,
            payload.customer_id,
            payload.role,
            payload.message,
            payload.timestamp,
        ],
    )
    con.close()
    # TODO: call mood-ml's internal inference endpoint and persist the
    # resulting score into mood_scores.
    return {"status": "ok"}


@router.get("/customer/{customer_id}/mood", response_model=MoodResponse)
def get_customer_mood(customer_id: str) -> MoodResponse:
    con = get_connection()
    row = con.execute(
        "SELECT customer_id, score, scale, model_version, computed_at "
        "FROM mood_scores WHERE customer_id = ?",
        [customer_id],
    ).fetchone()
    con.close()
    if row is None:
        raise HTTPException(status_code=404, detail="customer has no mood data yet")
    return MoodResponse(
        customer_id=row[0],
        score=row[1],
        scale=row[2],
        model_version=row[3],
        computed_at=row[4],
    )
