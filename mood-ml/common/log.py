"""Structured logging: one JSON object per line on stderr (see specs/10-common.md, CM-R02)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def now_iso() -> str:
    return to_iso(datetime.now(timezone.utc))


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": now_iso(), "level": record.levelname, "event": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "fields", None)
        suffix = f" {extra}" if extra else ""
        return f"{record.levelname:<7} {record.getMessage()}{suffix}"


def configure_logging(logger: logging.Logger, log_format: str, log_level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False


def log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """`run_id` goes first when present and is omitted when None (infer has no run_id)."""
    run_id = fields.pop("run_id", None)
    payload = {"run_id": run_id, **fields} if run_id is not None else fields
    logger.log(level, event, extra={"fields": payload})
