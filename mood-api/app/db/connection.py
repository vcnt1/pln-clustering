from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mood.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def init_db() -> None:
    con = get_connection()
    try:
        con.execute("BEGIN TRANSACTION")
        legacy = con.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'messages' AND column_name = 'id'"
        ).fetchone() is not None
        if legacy:
            con.execute("ALTER TABLE messages RENAME TO messages_legacy")
            con.execute("ALTER TABLE mood_scores RENAME TO mood_scores_legacy")
        _create_schema(con)
        if legacy:
            con.execute(
                "INSERT INTO customers SELECT customer_id, min(timestamp), max(timestamp) "
                "FROM messages_legacy GROUP BY customer_id"
            )
            con.execute(
                "INSERT INTO conversations "
                "SELECT conversation_id, customer_id, min(timestamp), max(timestamp), 'open', NULL "
                "FROM messages_legacy GROUP BY conversation_id, customer_id"
            )
            con.execute(
                "INSERT INTO messages "
                "SELECT id, conversation_id, customer_id, role, message, timestamp, timestamp "
                "FROM messages_legacy"
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _create_schema(con: duckdb.DuckDBPyConnection) -> None:

    # Dimensions: mutable state, upserted on ingest (data-model.md §3.1-3.2).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id VARCHAR PRIMARY KEY,
            first_seen_at TIMESTAMPTZ NOT NULL,
            last_message_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id VARCHAR PRIMARY KEY,
            customer_id VARCHAR NOT NULL,
            started_at TIMESTAMPTZ NOT NULL,
            last_message_at TIMESTAMPTZ NOT NULL,
            status VARCHAR NOT NULL DEFAULT 'open',
            closed_at TIMESTAMPTZ
        )
        """
    )

    # Facts: append-only, no UPDATE/DELETE (data-model.md §3.3-3.6, P2).
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            message_id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR NOT NULL,
            customer_id VARCHAR NOT NULL,
            role VARCHAR NOT NULL,
            text VARCHAR NOT NULL,
            sent_at TIMESTAMPTZ NOT NULL,
            received_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    # ADR-0007: builds the 30-customer-message window without a full scan.
    con.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_conversation_role_sent "
        "ON messages (conversation_id, role, sent_at)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS ix_messages_conversation_sent "
        "ON messages (conversation_id, sent_at)"
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mood_scores (
            mood_id VARCHAR PRIMARY KEY,
            request_id VARCHAR NOT NULL,
            customer_id VARCHAR NOT NULL,
            conversation_id VARCHAR NOT NULL,
            trigger_message_id VARCHAR NOT NULL,
            score DOUBLE NOT NULL CHECK (score BETWEEN -1.0 AND 1.0),
            scale VARCHAR NOT NULL,
            mood_label VARCHAR,
            model_version VARCHAR NOT NULL,
            computed_at TIMESTAMPTZ NOT NULL,
            persisted_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_failures (
            failure_id VARCHAR PRIMARY KEY,
            request_id VARCHAR NOT NULL,
            trigger_message_id VARCHAR NOT NULL,
            conversation_id VARCHAR NOT NULL,
            model_version VARCHAR,
            error_code VARCHAR NOT NULL,
            detail VARCHAR,
            occurred_at TIMESTAMPTZ NOT NULL
        )
        """
    )

    # Current mood is derived, never stored (P2) — data-model.md §3.7.
    con.execute(
        """
        CREATE OR REPLACE VIEW v_conversation_mood_latest AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY conversation_id ORDER BY computed_at DESC, persisted_at DESC
            ) AS rn
            FROM mood_scores
        ) WHERE rn = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE VIEW v_customer_mood_latest AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY customer_id ORDER BY computed_at DESC, persisted_at DESC
            ) AS rn
            FROM mood_scores
        ) WHERE rn = 1
        """
    )
