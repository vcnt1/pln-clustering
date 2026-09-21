"""[2] T4: builds the label set (one row per customer message) from the raw corpus (see specs/03-labels.md)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ingest.validate import (
    IngestGateError,
    check_ingest_gate,
    load_corpus,
    sha256_and_size,
)

VALID_LABELS = {-1.0, -0.5, 0.0, 0.5, 1.0}
REPORT_VERSION = "lb-1"
IO_RETRY_BACKOFF_SECONDS = (1, 2, 4)

logger = logging.getLogger("labels.build")

_GATE_REASON_TO_CODE = {
    "missing_report": "LB_R01_GATE_MISSING_REPORT",
    "status_not_ok": "LB_R01_GATE_STATUS_NOT_OK",
    "sha256_mismatch": "LB_R02_GATE_SHA256_MISMATCH",
}


class LabelGateError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class LabelSanityError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


class LabelIOError(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _with_io_retry(operation: str, func: Any, run_id: str) -> Any:
    """Retries an I/O callable 3x with 1s/2s/4s backoff on OSError (LB-R14)."""
    last_exc: OSError | None = None
    for attempt, backoff in enumerate((0, *IO_RETRY_BACKOFF_SECONDS), start=1):
        if backoff:
            time.sleep(backoff)
        try:
            return func()
        except OSError as exc:
            last_exc = exc
            _log(logging.WARNING, "io_retry", run_id=run_id, attempt=attempt, operation=operation, errno=exc.errno)
    raise LabelIOError("LB_R14_IO_FAILED", f"{operation} failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# build_labels — F0+F1 (gate) -> F2 (load) -> F3 (filter+reshape) -> F4 (sanity)
# ---------------------------------------------------------------------------


def _build_label_rows(customer_messages: list, label_set_id: str) -> list[dict[str, Any]]:
    """F3: reshapes customer messages into label rows (LB-R05 to LB-R09).
    A separate, named function so the F4 sanity check (below) is a genuine
    regression guard, testable by monkeypatching this step in isolation."""
    return [
        {
            "label_set_id": label_set_id,
            "target_type": "message",
            "target_id": m.message_id,
            "label_score": m.generated_label,
            "scale": "-1 to 1",
            "label_source": "synthetic",
            "annotator": None,
            "labeled_at": m.sent_at,
        }
        for m in customer_messages
    ]


def build_labels(
    corpus_path: str | Path, report_dir: str | Path = Path("data/reports")
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Returns (label_rows, meta). Writes nothing; does not decide an exit
    code (see specs/03-labels.md — same separation as ingest.validate_corpus)."""
    corpus_path = Path(corpus_path)
    corpus_id = corpus_path.stem
    report_dir = Path(report_dir)

    try:
        ingest_report = check_ingest_gate(corpus_path, report_dir)
    except IngestGateError as exc:
        raise LabelGateError(_GATE_REASON_TO_CODE[exc.reason], exc.detail) from exc

    messages = load_corpus(corpus_path)
    customer_messages = [m for m in messages if m.role == "customer"]
    label_set_id = f"ls-{corpus_id}"

    rows = _build_label_rows(customer_messages, label_set_id)

    # F4 — sanity check, computed independently of how `rows` was built above.
    expected_count = sum(1 for m in messages if m.role == "customer")
    if len(rows) != expected_count:
        raise LabelSanityError(
            "LB_R10_ROW_COUNT_MISMATCH",
            f"{len(rows)} label rows != {expected_count} customer messages",
        )

    target_ids = [r["target_id"] for r in rows]
    if len(set(target_ids)) != len(target_ids):
        raise LabelSanityError("LB_R11_DUPLICATE_TARGET_ID", "duplicate target_id in label rows")

    for row in rows:
        if row["label_score"] not in VALID_LABELS:
            raise LabelSanityError(
                "LB_R11_LABEL_SCORE_OUT_OF_DOMAIN", f"label_score={row['label_score']!r} out of domain"
            )

    meta = {
        "corpus_id": corpus_id,
        "corpus_path": str(corpus_path),
        "corpus_sha256": ingest_report["source"]["sha256"],
        "ingest_report_path": str(report_dir / corpus_id / "validation_report.json"),
        "label_set_id": label_set_id,
        "messages_read": len(messages),
        "messages_customer": len(customer_messages),
        "messages_agent_excluded": len(messages) - len(customer_messages),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------


def _write_parquet_atomic(rows: list[dict[str, Any]], dest: Path, run_id: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    def _write() -> None:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, tmp)
        os.replace(tmp, dest)

    _with_io_retry("write_label_set", _write, run_id)


def _write_report_atomic(report: dict[str, Any], dest: Path, run_id: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    def _write() -> None:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, dest)

    _with_io_retry("write_label_report", _write, run_id)


# ---------------------------------------------------------------------------
# Logging (same JSON-per-line pattern as ingest.validate)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": _now_iso(), "level": record.levelname, "event": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "fields", None)
        suffix = f" {extra}" if extra else ""
        return f"{record.levelname:<7} {record.getMessage()}{suffix}"


def _configure_logging(log_format: str, log_level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if log_format == "json" else _TextFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False


def _log(level: int, event: str, **fields: Any) -> None:
    run_id = fields.pop("run_id", None)
    logger.log(level, event, extra={"fields": {"run_id": run_id, **fields}})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m labels.build")
    parser.add_argument("path", help="path to the approved corpus .jsonl file")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    corpus_path = Path(args.path)

    _log(logging.INFO, "build_started", run_id=run_id, corpus_path=str(corpus_path))

    try:
        rows, meta = build_labels(corpus_path, args.report_dir)
    except LabelGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except LabelSanityError as exc:
        _log(logging.ERROR, "sanity_check_failed", run_id=run_id, check=exc.code, detail=exc.detail)
        return 4
    except LabelIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    _log(
        logging.INFO,
        "gate_checked",
        run_id=run_id,
        ingest_report_path=meta["ingest_report_path"],
        sha256_match=True,
    )
    _log(logging.INFO, "corpus_loaded", run_id=run_id, messages_read=meta["messages_read"])
    _log(
        logging.INFO,
        "label_rows_built",
        run_id=run_id,
        messages_customer=meta["messages_customer"],
        messages_agent_excluded=meta["messages_agent_excluded"],
    )
    _log(
        logging.INFO,
        "sanity_check_passed",
        run_id=run_id,
        rows=len(rows),
        checks=["row_count", "label_domain", "target_id_unique"],
    )

    labels_dir = Path(args.labels_dir)
    report_dir = Path(args.report_dir)
    parquet_path = labels_dir / f"{meta['label_set_id']}.parquet"

    try:
        _write_parquet_atomic(rows, parquet_path, run_id)
        parquet_sha256, _ = sha256_and_size(parquet_path)

        label_levels = Counter(row["label_score"] for row in rows)
        finished_at = datetime.now(timezone.utc)
        report = {
            "report_version": REPORT_VERSION,
            "run": {
                "run_id": run_id,
                "started_at": started_at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{started_at.microsecond // 1000:03d}Z",
                "finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%S.") + f"{finished_at.microsecond // 1000:03d}Z",
                "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
                "tool": "labels.build",
                "python": sys.version.split()[0],
            },
            "source": {
                "corpus_id": meta["corpus_id"],
                "corpus_path": meta["corpus_path"],
                "corpus_sha256": meta["corpus_sha256"],
                "ingest_report_path": meta["ingest_report_path"],
            },
            "label_set": {
                "label_set_id": meta["label_set_id"],
                "path": str(parquet_path),
                "sha256": parquet_sha256,
                "rows": len(rows),
            },
            "counts": {
                "messages_read": meta["messages_read"],
                "messages_customer": meta["messages_customer"],
                "messages_agent_excluded": meta["messages_agent_excluded"],
            },
            "distributions": {
                "label_levels": {str(level): n for level, n in sorted(label_levels.items())},
            },
        }
        report_path = report_dir / meta["corpus_id"] / "label_report.json"
        _write_report_atomic(report, report_path, run_id)
    except LabelIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    _log(
        logging.INFO,
        "build_finished",
        run_id=run_id,
        label_set_id=meta["label_set_id"],
        rows=len(rows),
        duration_ms=report["run"]["duration_ms"],
        parquet_path=str(parquet_path),
        report_path=str(report_path),
    )
    print(str(parquet_path))
    print(str(report_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
