from pathlib import Path

import duckdb

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mood.duckdb"


def get_connection() -> duckdb.DuckDBPyConnection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH))


def init_db() -> None:
    con = get_connection()
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id VARCHAR PRIMARY KEY,
            conversation_id VARCHAR,
            customer_id VARCHAR,
            role VARCHAR,
            message VARCHAR,
            timestamp TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS mood_scores (
            customer_id VARCHAR PRIMARY KEY,
            score DOUBLE,
            scale VARCHAR,
            model_version VARCHAR,
            computed_at TIMESTAMP
        )
        """
    )
    con.close()
