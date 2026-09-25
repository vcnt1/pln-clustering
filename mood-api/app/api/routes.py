import uuid
from datetime import datetime, timezone
from typing import Optional

import duckdb
from fastapi import APIRouter, HTTPException, Response

from app.db.connection import get_connection
from app.ml_client import HistoryMessage, InferenceError, InferRequest, call_infer
from app.models.schemas import (
    ConversationMessage,
    ConversationSummary,
    IngestRequest,
    IngestResponse,
    MoodResponse,
)

router = APIRouter()

# README "Erros a cobrir": 413 threshold. Kept in sync with the synthetic
# corpus contract (mood-ml/specs/01-dataset-contract.md, max 1000 chars).
MAX_MESSAGE_LENGTH = 1000
# ADR-0004: 3 consecutive non-quarantine failures since the last success.
QUARANTINE_THRESHOLD = 3
# ADR-0007: last 30 `customer` messages of the conversation.
HISTORY_WINDOW = 30


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise HTTPException(status_code=400, detail="timestamp must include a timezone")
    return value.astimezone(timezone.utc)


@router.post("/ingest", response_model=IngestResponse)
def ingest_message(payload: IngestRequest) -> IngestResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    if len(payload.message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=413, detail=f"message exceeds {MAX_MESSAGE_LENGTH} characters"
        )

    sent_at = _to_utc(payload.timestamp)
    con = get_connection()
    try:
        existing_conversation = con.execute(
            "SELECT customer_id FROM conversations WHERE conversation_id = ?",
            [payload.conversation_id],
        ).fetchone()

        # ADR-0006: an agent never opens a conversation, only replies to one.
        if existing_conversation is None and payload.role == "agent":
            raise HTTPException(
                status_code=422, detail="an agent message cannot start a new conversation"
            )
        if existing_conversation is not None and existing_conversation[0] != payload.customer_id:
            raise HTTPException(
                status_code=400, detail="customer_id does not match the conversation owner"
            )

        message_id = str(uuid.uuid4())
        received_at = datetime.now(timezone.utc)

        _upsert_customer(con, payload.customer_id, sent_at)
        _upsert_conversation(
            con, payload.conversation_id, payload.customer_id, existing_conversation is None, sent_at
        )

        con.execute(
            "INSERT INTO messages "
            "(message_id, conversation_id, customer_id, role, text, sent_at, received_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                message_id,
                payload.conversation_id,
                payload.customer_id,
                payload.role,
                payload.message,
                sent_at,
                received_at,
            ],
        )

        # Only a customer message drives inference (README, data-model.md §2.1).
        if payload.role == "customer":
            _run_inference(con, payload.conversation_id, payload.customer_id, message_id)
    finally:
        con.close()

    return IngestResponse(status="ok", message_id=message_id)


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations() -> list[ConversationSummary]:
    con = get_connection()
    try:
        conversation_rows = con.execute(
            "SELECT conversation_id, customer_id, last_message_at FROM conversations "
            "ORDER BY last_message_at DESC"
        ).fetchall()

        summaries = []
        for conversation_id, customer_id, last_message_at in conversation_rows:
            message_rows = con.execute(
                "SELECT message_id, role, text, sent_at FROM messages "
                "WHERE conversation_id = ? ORDER BY sent_at ASC, received_at ASC",
                [conversation_id],
            ).fetchall()
            # data-model.md §7: scope the mood score to the owning conversation.
            mood_row = con.execute(
                "SELECT score, scale FROM v_conversation_mood_latest WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()

            summaries.append(
                ConversationSummary(
                    conversation_id=conversation_id,
                    customer_id=customer_id,
                    last_message_at=last_message_at,
                    score=mood_row[0] if mood_row else None,
                    scale=mood_row[1] if mood_row else None,
                    messages=[
                        ConversationMessage(message_id=row[0], role=row[1], text=row[2], sent_at=row[3])
                        for row in message_rows
                    ],
                )
            )
        return summaries
    finally:
        con.close()


@router.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: str) -> Response:
    con = get_connection()
    try:
        existing = con.execute(
            "SELECT customer_id FROM conversations WHERE conversation_id = ?", [conversation_id]
        ).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        customer_id = existing[0]

        try:
            con.execute("BEGIN TRANSACTION")
            # Facts are append-only elsewhere (P2), but a full conversation delete
            # must cascade to keep DuckDB free of orphaned rows.
            con.execute("DELETE FROM inference_failures WHERE conversation_id = ?", [conversation_id])
            con.execute("DELETE FROM mood_scores WHERE conversation_id = ?", [conversation_id])
            con.execute("DELETE FROM messages WHERE conversation_id = ?", [conversation_id])
            con.execute("DELETE FROM conversations WHERE conversation_id = ?", [conversation_id])
            remaining = con.execute(
                "SELECT count(*) FROM conversations WHERE customer_id = ?", [customer_id]
            ).fetchone()[0]
            if remaining == 0:
                con.execute("DELETE FROM customers WHERE customer_id = ?", [customer_id])
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    finally:
        con.close()
    return Response(status_code=204)


@router.get("/customer/{customer_id}/mood", response_model=MoodResponse)
def get_customer_mood(customer_id: str) -> MoodResponse:
    con = get_connection()
    try:
        row = con.execute(
            "SELECT customer_id, conversation_id, score, scale, mood_label, model_version, computed_at "
            "FROM v_customer_mood_latest WHERE customer_id = ?",
            [customer_id],
        ).fetchone()
    finally:
        con.close()

    # ADR-0004: 404 only when there is no successful inference at all — never
    # a neutral/placeholder score.
    if row is None:
        raise HTTPException(status_code=404, detail="no mood computed for this customer yet")

    return MoodResponse(
        customer_id=row[0],
        conversation_id=row[1],
        score=row[2],
        scale=row[3],
        mood_label=row[4],
        model_version=row[5],
        computed_at=row[6],
    )


def _upsert_customer(
    con: duckdb.DuckDBPyConnection, customer_id: str, sent_at: datetime
) -> None:
    existing = con.execute(
        "SELECT customer_id FROM customers WHERE customer_id = ?", [customer_id]
    ).fetchone()
    if existing is None:
        con.execute(
            "INSERT INTO customers (customer_id, first_seen_at, last_message_at) VALUES (?, ?, ?)",
            [customer_id, sent_at, sent_at],
        )
    else:
        con.execute(
            "UPDATE customers SET last_message_at = ? WHERE customer_id = ?",
            [sent_at, customer_id],
        )


def _upsert_conversation(
    con: duckdb.DuckDBPyConnection, conversation_id: str, customer_id: str, is_new: bool, sent_at: datetime
) -> None:
    if is_new:
        con.execute(
            "INSERT INTO conversations "
            "(conversation_id, customer_id, started_at, last_message_at, status, closed_at) "
            "VALUES (?, ?, ?, ?, 'open', NULL)",
            [conversation_id, customer_id, sent_at, sent_at],
        )
    else:
        con.execute(
            "UPDATE conversations SET last_message_at = ? WHERE conversation_id = ?",
            [sent_at, conversation_id],
        )


def _run_inference(
    con: duckdb.DuckDBPyConnection, conversation_id: str, customer_id: str, trigger_message_id: str
) -> None:
    request_id = str(uuid.uuid4())

    quarantine_count = con.execute(
        """
        SELECT count(*) FROM inference_failures
        WHERE conversation_id = ?
          AND error_code <> 'quarantined'
          AND occurred_at > coalesce(
                (SELECT max(persisted_at) FROM mood_scores WHERE conversation_id = ?),
                TIMESTAMP '-infinity'
              )
        """,
        [conversation_id, conversation_id],
    ).fetchone()[0]

    # ADR-0004: past the threshold, stop calling mood-ml until an operator clears it.
    if quarantine_count >= QUARANTINE_THRESHOLD:
        _record_failure(
            con,
            request_id,
            trigger_message_id,
            conversation_id,
            None,
            "quarantined",
            "conversation quarantined after repeated inference failures",
        )
        return

    history_rows = con.execute(
        """
        SELECT message_id, role, text FROM messages
        WHERE conversation_id = ? AND role = 'customer' AND message_id <> ?
        ORDER BY sent_at DESC, received_at DESC
        LIMIT ?
        """,
        [conversation_id, trigger_message_id, HISTORY_WINDOW - 1],
    ).fetchall()
    history_rows.reverse()
    trigger_row = con.execute(
        "SELECT message_id, role, text FROM messages WHERE message_id = ?",
        [trigger_message_id],
    ).fetchone()
    history_rows.append(trigger_row)

    history = [
        HistoryMessage(message_id=row[0], role=row[1], text=row[2])
        for row in history_rows
    ]

    infer_request = InferRequest(
        request_id=request_id,
        customer_id=customer_id,
        conversation_id=conversation_id,
        history=history,
    )

    try:
        result = call_infer(infer_request)
    except InferenceError as exc:
        _record_failure(con, request_id, trigger_message_id, conversation_id, None, exc.error_code, exc.detail)
        return

    if not (-1.0 <= result.score <= 1.0):
        _record_failure(
            con,
            request_id,
            trigger_message_id,
            conversation_id,
            result.model_version,
            "score_out_of_range",
            f"score {result.score} outside [-1.0, 1.0]",
        )
        return

    con.execute(
        "INSERT INTO mood_scores "
        "(mood_id, request_id, customer_id, conversation_id, trigger_message_id, score, scale, "
        "mood_label, model_version, computed_at, persisted_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(uuid.uuid4()),
            request_id,
            customer_id,
            conversation_id,
            trigger_message_id,
            result.score,
            result.scale,
            result.mood_label,
            result.model_version,
            result.computed_at,
            datetime.now(timezone.utc),
        ],
    )


def _record_failure(
    con: duckdb.DuckDBPyConnection,
    request_id: str,
    trigger_message_id: str,
    conversation_id: str,
    model_version: Optional[str],
    error_code: str,
    detail: str,
) -> None:
    con.execute(
        "INSERT INTO inference_failures "
        "(failure_id, request_id, trigger_message_id, conversation_id, model_version, error_code, detail, occurred_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            str(uuid.uuid4()),
            request_id,
            trigger_message_id,
            conversation_id,
            model_version,
            error_code,
            detail,
            datetime.now(timezone.utc),
        ],
    )
