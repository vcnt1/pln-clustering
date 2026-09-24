"""[1] Validates a dc-1 .jsonl corpus and writes validation_report.json (see specs/02-ingest-validate.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from common.errors import PipelineError
from common.io import with_io_retry, write_json_atomic
from common.log import configure_logging, log, to_iso
from transform.mask import count_pii

# ---------------------------------------------------------------------------
# Constants (dc-1, spec 01; ig-1, spec 02)
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "corpus_id",
    "conversation_id",
    "customer_id",
    "message_id",
    "role",
    "text",
    "sent_at",
    "persona",
    "generated_label",
}
VALID_ROLES = {"customer", "agent"}
VALID_LABELS = {-1.0, -0.5, 0.0, 0.5, 1.0}
MAX_TEXT_LEN = 1000
SNAKE_CASE_RE = re.compile(r"^[a-z]+(?:_[a-z]+)*$")
SENT_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ID_PREFIX = "syn-"

MIN_CUSTOMERS = 150
MIN_CONVERSATIONS = 300
MIN_CUSTOMER_MESSAGES = 2000
MIN_LEVEL_PCT = 0.10
MIN_PERSONAS = 4
MIN_PERSONA_PCT = 0.10
PII_RATIO_MIN = 0.05
PII_RATIO_MAX = 0.15
CONV_LEN_MIN = 4
CONV_LEN_MAX = 20
MOOD_VARIATION_MIN_PCT = 0.30
NEAR_LIMIT_MARGIN = 0.02  # warn within 2 points of a DC-R10/R11/R14 floor/ceiling

MAX_SCHEMA_ERRORS = 50

REPORT_VERSION = "ig-1"
CONTRACT_VERSION = "dc-1"

logger = logging.getLogger("ingest.validate")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CorpusPathError(PipelineError):
    """F0 usage error (IG-R01/IG-R02). Never becomes part of a report."""


class CorpusIOError(PipelineError):
    """I/O failure after exhausting retries (IG-R15)."""


_log = partial(log, logger)
_configure_logging = partial(configure_logging, logger)
_with_io_retry = partial(with_io_retry, logger=logger, error=lambda detail: CorpusIOError("IG_R15_IO_FAILED", detail))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Message:
    corpus_id: str
    conversation_id: str
    customer_id: str
    message_id: str
    role: str
    text: str
    sent_at: str
    persona: str
    generated_label: float | None


@dataclass
class ValidationReport:
    report_version: str
    contract_version: str
    status: str
    run: dict[str, Any]
    source: dict[str, Any]
    counts: dict[str, Any]
    distributions: dict[str, Any]
    realism: dict[str, Any]
    manual_checks: dict[str, Any]
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    errors_truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# F0 — path/usage checks
# ---------------------------------------------------------------------------


def _check_path_usage(path: Path) -> str:
    """Validates the CLI path argument. Returns the expected corpus_id
    (filename stem). Raises CorpusPathError for IG-R01/IG-R02 violations."""
    if path.suffix != ".jsonl":
        raise CorpusPathError("IG_R01_NOT_READABLE", f"expected a .jsonl file: {path}")
    if not path.exists():
        raise CorpusPathError("IG_R01_FILE_NOT_FOUND", f"file not found: {path}")
    if not path.is_file() or not os.access(path, os.R_OK):
        raise CorpusPathError("IG_R01_NOT_READABLE", f"file not readable: {path}")
    if path.stat().st_size == 0:
        raise CorpusPathError("IG_R01_EMPTY_FILE", f"file is empty: {path}")

    resolved = path.resolve()
    if not (resolved.parent.name == "synthetic" and resolved.parent.parent.name == "raw"):
        raise CorpusPathError(
            "IG_R02_PATH_OUTSIDE_SYNTHETIC",
            f"path must be under data/raw/synthetic/: {path}",
        )
    return path.stem


# ---------------------------------------------------------------------------
# F1 — fingerprint
# ---------------------------------------------------------------------------


def sha256_and_size(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


# ---------------------------------------------------------------------------
# check_ingest_gate — shared portão, reused by labels/build.py and
# transform/split.py (LB-R01/R02, SP-R01) so the "is this corpus approved?"
# check has exactly one implementation.
# ---------------------------------------------------------------------------


class IngestGateError(Exception):
    """The ingest gate (validation_report.json) does not approve this corpus.
    `.reason` is one of "missing_report" | "status_not_ok" | "sha256_mismatch";
    callers translate it into their own spec-mandated error code."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def check_ingest_gate(
    corpus_path: str | Path, report_dir: str | Path = Path("data/reports")
) -> dict[str, Any]:
    """Confirms a validation_report.json with status="ok" and a matching
    sha256 exists for `corpus_path`. Returns the parsed report dict. Raises
    IngestGateError otherwise, without reading the corpus's content."""
    corpus_path = Path(corpus_path)
    corpus_id = corpus_path.stem
    report_path = Path(report_dir) / corpus_id / "validation_report.json"
    if not report_path.exists():
        raise IngestGateError("missing_report", f"no validation_report.json for {corpus_id}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "ok":
        raise IngestGateError(
            "status_not_ok", f"validation_report.json status is {report.get('status')!r}"
        )

    sha256_hex, _ = sha256_and_size(corpus_path)
    if report["source"]["sha256"] != sha256_hex:
        raise IngestGateError("sha256_mismatch", "corpus file changed after validation")

    return report


# ---------------------------------------------------------------------------
# load_corpus — canonical ordered reader (IG-R13)
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path) -> list[Message]:
    """Reads a dc-1 .jsonl corpus and returns messages ordered by
    (conversation_id, sent_at). Assumes the corpus already passed
    validate_corpus(); does not re-validate."""
    path = Path(path)
    messages: list[Message] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            messages.append(Message(**{k: row[k] for k in REQUIRED_FIELDS}))
    messages.sort(key=lambda m: (m.conversation_id, m.sent_at))
    return messages


# ---------------------------------------------------------------------------
# validate_corpus — F0 -> F1 -> F3 -> F4 -> F5
# ---------------------------------------------------------------------------


def validate_corpus(path: str | Path, skip_composition: bool = False) -> ValidationReport:
    """Validates a dc-1 corpus and returns a ValidationReport. Does not write
    any file, decide an exit code, or perform the idempotency check — those
    are the CLI's responsibility (see specs/02-ingest-validate.md §3.1a)."""
    path = Path(path)
    started_at = datetime.now(timezone.utc)
    corpus_id = _check_path_usage(path)
    sha256_hex, size_bytes = _with_io_retry("open_corpus", lambda: sha256_and_size(path))

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors_truncated = False

    def add_error(code: str, line: int | None, message_id: str | None, detail: str) -> None:
        nonlocal errors_truncated
        if len(errors) >= MAX_SCHEMA_ERRORS:
            errors_truncated = True
            return
        errors.append({"code": code, "line": line, "message_id": message_id, "detail": detail})

    # -- accumulators --------------------------------------------------
    lines_read = 0
    lines_blank_skipped = 0
    seen_message_ids: set[str] = set()
    conv_customer: dict[str, str] = {}
    customer_persona: dict[str, str] = {}
    conv_sent_ats: dict[str, set[str]] = {}
    conv_last_sent_at: dict[str, str] = {}
    conv_roles: dict[str, set[str]] = {}
    conv_message_count: dict[str, int] = {}
    customer_ids: set[str] = set()
    conversation_ids: set[str] = set()
    persona_customers: dict[str, set[str]] = {}
    customer_message_count = 0
    agent_message_count = 0
    label_counts: dict[float, int] = dict.fromkeys(VALID_LABELS, 0)
    conv_labels: dict[str, set[float]] = {}
    pii_counts = {"cpf": 0, "phone": 0, "email": 0}
    customer_messages_with_pii = 0
    file_unordered = False

    with path.open("rb") as fb:
        has_bom = fb.read(3) == b"\xef\xbb\xbf"
    if has_bom:
        add_error("IG_R04_ENCODING_INVALID", None, None, "file has a UTF-8 BOM")

    encoding_ok = not has_bom
    with _with_io_retry("open_corpus", lambda: path.open("r", encoding="utf-8")) as f:
        line_no = 0
        while encoding_ok:
            line_no += 1
            try:
                raw_line = next(f)
            except StopIteration:
                break
            except UnicodeDecodeError as exc:
                add_error("IG_R04_ENCODING_INVALID", line_no, None, f"invalid UTF-8: {exc}")
                encoding_ok = False
                break

            line = raw_line.rstrip("\n").rstrip("\r")
            if not line.strip():
                lines_blank_skipped += 1
                continue
            lines_read += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                add_error("DC_R01_INVALID_JSON", line_no, None, str(exc))
                continue

            violation = _schema_violation(row, corpus_id)
            if violation:
                add_error("DC_R01_SCHEMA_VIOLATION", line_no, row.get("message_id"), violation)
                continue

            message_id = row["message_id"]
            conv_id = row["conversation_id"]
            cust_id = row["customer_id"]
            persona = row["persona"]
            role = row["role"]
            sent_at = row["sent_at"]
            label = row["generated_label"]

            if row["corpus_id"] != corpus_id:
                add_error(
                    "IG_R03_CORPUS_ID_INCONSISTENT",
                    line_no,
                    message_id,
                    f"corpus_id {row['corpus_id']!r} != expected {corpus_id!r}",
                )
                continue

            prefix_bad = any(
                not str(v).startswith(ID_PREFIX) for v in (conv_id, cust_id, message_id)
            )
            if prefix_bad:
                add_error("DC_R03_ID_PREFIX_INVALID", line_no, message_id, "missing 'syn-' prefix")
                continue

            if role == "customer":
                if label not in VALID_LABELS:
                    add_error(
                        "DC_R07_LABEL_MISSING_OR_INVALID",
                        line_no,
                        message_id,
                        f"generated_label={label!r} invalid for role=customer",
                    )
                    continue
            elif label is not None:
                add_error(
                    "DC_R08_AGENT_LABEL_NOT_NULL",
                    line_no,
                    message_id,
                    f"generated_label={label!r} must be null for role=agent",
                )
                continue

            # -- F4: cross-line integrity -------------------------------
            if message_id in seen_message_ids:
                add_error("DC_R02_DUPLICATE_MESSAGE_ID", line_no, message_id, "duplicate message_id")
                continue
            seen_message_ids.add(message_id)

            existing_customer = conv_customer.get(conv_id)
            if existing_customer is not None and existing_customer != cust_id:
                add_error(
                    "DC_R04_CONVERSATION_MULTI_CUSTOMER",
                    line_no,
                    message_id,
                    f"conversation {conv_id} has customers {existing_customer!r} and {cust_id!r}",
                )
                continue
            conv_customer[conv_id] = cust_id

            existing_persona = customer_persona.get(cust_id)
            if existing_persona is not None and existing_persona != persona:
                add_error(
                    "DC_R05_CUSTOMER_MULTI_PERSONA",
                    line_no,
                    message_id,
                    f"customer {cust_id} has personas {existing_persona!r} and {persona!r}",
                )
                continue
            customer_persona[cust_id] = persona

            conv_sent_at_set = conv_sent_ats.setdefault(conv_id, set())
            if sent_at in conv_sent_at_set:
                add_error(
                    "DC_R06_SENT_AT_DUPLICATE",
                    line_no,
                    message_id,
                    f"duplicate sent_at {sent_at!r} in conversation {conv_id}",
                )
                continue
            conv_sent_at_set.add(sent_at)

            last_sent_at = conv_last_sent_at.get(conv_id)
            if last_sent_at is not None and sent_at < last_sent_at:
                file_unordered = True
            conv_last_sent_at[conv_id] = sent_at

            # -- accumulate for F5 --------------------------------------
            customer_ids.add(cust_id)
            conversation_ids.add(conv_id)
            conv_roles.setdefault(conv_id, set()).add(role)
            conv_message_count[conv_id] = conv_message_count.get(conv_id, 0) + 1
            persona_customers.setdefault(persona, set()).add(cust_id)

            if role == "customer":
                customer_message_count += 1
                label_counts[label] += 1
                conv_labels.setdefault(conv_id, set()).add(label)
                pii = count_pii(row["text"])
                if any(pii.values()):
                    customer_messages_with_pii += 1
                for kind, n in pii.items():
                    pii_counts[kind] += n
            else:
                agent_message_count += 1

    if file_unordered:
        warnings.append(
            {"code": "file_unordered", "detail": "lines are not in (conversation_id, sent_at) order"}
        )
    if lines_blank_skipped:
        warnings.append(
            {"code": "blank_lines_skipped", "detail": f"{lines_blank_skipped} blank line(s) skipped"}
        )

    schema_or_integrity_failed = len(errors) > 0

    counts = {
        "messages": lines_read,
        "messages_customer": customer_message_count,
        "messages_agent": agent_message_count,
        "conversations": len(conversation_ids),
        "customers": len(customer_ids),
        "personas": len(persona_customers),
    }

    distributions: dict[str, Any] = {}
    realism: dict[str, Any] = {}
    composition_status = "not_evaluated"

    if not schema_or_integrity_failed:
        composition_status = "skipped" if skip_composition else "evaluated"
        distributions, realism, composition_errors, composition_warnings = _evaluate_composition(
            customer_message_count=customer_message_count,
            conversation_ids=conversation_ids,
            customer_ids=customer_ids,
            label_counts=label_counts,
            persona_customers=persona_customers,
            conv_message_count=conv_message_count,
            conv_roles=conv_roles,
            conv_labels=conv_labels,
            pii_counts=pii_counts,
            customer_messages_with_pii=customer_messages_with_pii,
            skip_composition=skip_composition,
        )
        for err in composition_errors:
            add_error(*err)
        warnings.extend(composition_warnings)

    status = "rejected" if errors else "ok"

    finished_at = datetime.now(timezone.utc)
    run = {
        "run_id": uuid.uuid4().hex[:8],
        "started_at": to_iso(started_at),
        "finished_at": to_iso(finished_at),
        "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
        "tool": "ingest.validate",
        "python": sys.version.split()[0],
        "flags": {"skip_composition": skip_composition},
    }
    source = {
        "corpus_id": corpus_id,
        "path": str(path),
        "sha256": sha256_hex,
        "bytes": size_bytes,
        "lines_read": lines_read,
        "lines_blank_skipped": lines_blank_skipped,
    }
    manual_checks = {
        "DC-R12": {"state": "pending", "sample_path": None},
        "DC-R13": {"state": "declared_by_location"},
    }
    if composition_status != "evaluated":
        distributions = distributions or {}
        realism = {**realism, "composition": composition_status} if realism else {"composition": composition_status}

    return ValidationReport(
        report_version=REPORT_VERSION,
        contract_version=CONTRACT_VERSION,
        status=status,
        run=run,
        source=source,
        counts=counts,
        distributions=distributions,
        realism=realism,
        manual_checks=manual_checks,
        errors=errors,
        warnings=warnings,
        errors_truncated=errors_truncated,
    )


def _schema_violation(row: dict[str, Any], expected_corpus_id: str) -> str | None:
    if not isinstance(row, dict):
        return "line is not a JSON object"
    missing = REQUIRED_FIELDS - row.keys()
    extra = row.keys() - REQUIRED_FIELDS
    if missing:
        return f"missing fields: {sorted(missing)}"
    if extra:
        return f"unexpected fields: {sorted(extra)}"

    for field_name in (
        "corpus_id",
        "conversation_id",
        "customer_id",
        "message_id",
        "role",
        "text",
        "sent_at",
        "persona",
    ):
        if not isinstance(row[field_name], str):
            return f"field {field_name!r} must be a string"

    if row["role"] not in VALID_ROLES:
        return f"role {row['role']!r} not in {sorted(VALID_ROLES)}"

    text = row["text"]
    if not text.strip():
        return "text is empty after strip()"
    if len(text) > MAX_TEXT_LEN:
        return f"text exceeds {MAX_TEXT_LEN} characters"

    if not SENT_AT_RE.match(row["sent_at"]):
        return f"sent_at {row['sent_at']!r} is not ISO-8601 UTC with 'Z' suffix"

    if not SNAKE_CASE_RE.match(row["persona"]):
        return f"persona {row['persona']!r} is not snake_case"

    label = row["generated_label"]
    if label is not None and not isinstance(label, (int, float)):
        return "generated_label must be a number or null"

    return None


def _evaluate_composition(
    *,
    customer_message_count: int,
    conversation_ids: set[str],
    customer_ids: set[str],
    label_counts: dict[float, int],
    persona_customers: dict[str, set[str]],
    conv_message_count: dict[str, int],
    conv_roles: dict[str, set[str]],
    conv_labels: dict[str, set[float]],
    pii_counts: dict[str, int],
    customer_messages_with_pii: int,
    skip_composition: bool,
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[str, None, None, str]], list[dict[str, Any]]]:
    errors: list[tuple[str, None, None, str]] = []
    warnings: list[dict[str, Any]] = []

    if not skip_composition:
        if len(customer_ids) < MIN_CUSTOMERS:
            errors.append(
                ("DC_R09_COMPOSITION_BELOW_MINIMUM", None, None, f"{len(customer_ids)} customers < {MIN_CUSTOMERS}")
            )
        if len(conversation_ids) < MIN_CONVERSATIONS:
            errors.append(
                (
                    "DC_R09_COMPOSITION_BELOW_MINIMUM",
                    None,
                    None,
                    f"{len(conversation_ids)} conversations < {MIN_CONVERSATIONS}",
                )
            )
        if customer_message_count < MIN_CUSTOMER_MESSAGES:
            errors.append(
                (
                    "DC_R09_COMPOSITION_BELOW_MINIMUM",
                    None,
                    None,
                    f"{customer_message_count} customer messages < {MIN_CUSTOMER_MESSAGES}",
                )
            )

        for level, n in label_counts.items():
            pct = n / customer_message_count if customer_message_count else 0.0
            if pct < MIN_LEVEL_PCT:
                errors.append(
                    ("DC_R10_LABEL_LEVEL_UNDERREPRESENTED", None, None, f"level {level} at {pct:.3f} < {MIN_LEVEL_PCT}")
                )
            elif pct < MIN_LEVEL_PCT + NEAR_LIMIT_MARGIN:
                warnings.append(
                    {"code": "threshold_near_limit", "rule": "DC-R10", "observed": pct, "limit": MIN_LEVEL_PCT}
                )

        if len(persona_customers) < MIN_PERSONAS:
            errors.append(
                ("DC_R11_PERSONA_UNDERREPRESENTED", None, None, f"{len(persona_customers)} personas < {MIN_PERSONAS}")
            )
        for persona, custs in persona_customers.items():
            pct = len(custs) / len(customer_ids) if customer_ids else 0.0
            if pct < MIN_PERSONA_PCT:
                errors.append(
                    ("DC_R11_PERSONA_UNDERREPRESENTED", None, None, f"persona {persona} at {pct:.3f} < {MIN_PERSONA_PCT}")
                )
            elif pct < MIN_PERSONA_PCT + NEAR_LIMIT_MARGIN:
                warnings.append(
                    {"code": "threshold_near_limit", "rule": "DC-R11", "observed": pct, "limit": MIN_PERSONA_PCT}
                )

    pii_ratio = customer_messages_with_pii / customer_message_count if customer_message_count else 0.0
    if not (PII_RATIO_MIN <= pii_ratio <= PII_RATIO_MAX):
        errors.append(
            ("DC_R14_PII_RATIO_OUT_OF_RANGE", None, None, f"pii_ratio {pii_ratio:.3f} outside [{PII_RATIO_MIN}, {PII_RATIO_MAX}]")
        )
    elif pii_ratio < PII_RATIO_MIN + NEAR_LIMIT_MARGIN or pii_ratio > PII_RATIO_MAX - NEAR_LIMIT_MARGIN:
        warnings.append({"code": "threshold_near_limit", "rule": "DC-R14", "observed": pii_ratio, "limit": None})

    for conv_id, n in conv_message_count.items():
        if not (CONV_LEN_MIN <= n <= CONV_LEN_MAX):
            errors.append(
                ("DC_R15_CONVERSATION_LENGTH_INVALID", None, None, f"conversation {conv_id} has {n} messages")
            )
        if conv_roles.get(conv_id) != VALID_ROLES:
            errors.append(
                ("DC_R15_CONVERSATION_SINGLE_ROLE", None, None, f"conversation {conv_id} missing a role")
            )

    varied = sum(1 for labels in conv_labels.values() if len(labels) >= 2)
    variation_ratio = varied / len(conversation_ids) if conversation_ids else 0.0
    if variation_ratio < MOOD_VARIATION_MIN_PCT:
        errors.append(
            ("DC_R16_MOOD_VARIATION_INSUFFICIENT", None, None, f"mood_variation_ratio {variation_ratio:.3f} < {MOOD_VARIATION_MIN_PCT}")
        )

    lengths = sorted(conv_message_count.values())
    distributions = {
        "label_levels": {
            str(level): {
                "n": n,
                "pct": (n / customer_message_count if customer_message_count else 0.0),
            }
            for level, n in sorted(label_counts.items())
        },
        "personas": {
            persona: {
                "customers": len(custs),
                "pct": (len(custs) / len(customer_ids) if customer_ids else 0.0),
            }
            for persona, custs in sorted(persona_customers.items())
        },
        "conversation_length": {
            "min": lengths[0] if lengths else 0,
            "p50": lengths[len(lengths) // 2] if lengths else 0,
            "max": lengths[-1] if lengths else 0,
        },
    }
    realism = {
        "pii_ratio": pii_ratio,
        "pii_by_kind": pii_counts,
        "mood_variation_ratio": variation_ratio,
    }
    return distributions, realism, errors, warnings


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------


def _write_report_atomic(report: dict[str, Any], dest: Path) -> None:
    _with_io_retry("write_report", lambda: write_json_atomic(report, dest))


def _write_review_sample(path: Path, dest: Path, n: int, seed: int) -> Path:
    messages = load_corpus(path)
    customer_msgs = [m for m in messages if m.role == "customer"]
    rng = random.Random(seed)
    sample = rng.sample(customer_msgs, k=min(n, len(customer_msgs)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for m in sample:
            f.write(json.dumps(asdict(m), ensure_ascii=False) + "\n")
    return dest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ingest.validate")
    parser.add_argument("path", help="path to the corpus .jsonl file")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--skip-composition", action="store_true")
    parser.add_argument("--review-sample", type=int, default=None, metavar="N")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    path = Path(args.path)

    flags = {
        "skip_composition": args.skip_composition,
        "review_sample": args.review_sample,
        "log_format": args.log_format,
        "log_level": args.log_level,
    }
    _log(logging.INFO, "ingest_started", run_id=run_id, path=str(path), flags=flags)

    try:
        corpus_id = _check_path_usage(path)
    except CorpusPathError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        print(f"{exc.code}: {exc.detail}", file=sys.stderr)
        return 2

    report_dir = Path(args.report_dir) / corpus_id
    report_path = report_dir / "validation_report.json"

    try:
        sha256_hex, _ = _with_io_retry("open_corpus", lambda: sha256_and_size(path))
    except CorpusIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, operation="fingerprint", detail=exc.detail)
        return 1

    _log(logging.INFO, "fingerprint_computed", run_id=run_id, sha256=sha256_hex)

    existing: dict[str, Any] | None = None
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))

    if existing is not None and existing["source"]["sha256"] == sha256_hex:
        if existing["status"] == "ok":
            _log(logging.INFO, "already_validated", run_id=run_id, sha256=sha256_hex)
            return 0
        # status == "rejected" with the same content: fall through and revalidate.
    elif existing is not None:
        _log(
            logging.ERROR,
            "gate_blocked",
            run_id=run_id,
            reason="hash_conflict",
            existing_sha256=existing["source"]["sha256"],
            computed_sha256=sha256_hex,
        )
        print(
            f"IG_R11_HASH_CONFLICT: {report_path} already describes a different "
            f"corpus; use a new corpus_id (dc-1 §3: syn-AAAA-MM-DD-<letra>)",
            file=sys.stderr,
        )
        return 4

    try:
        report = validate_corpus(path, skip_composition=args.skip_composition)
    except CorpusIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, operation="validate", detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    report.run["run_id"] = run_id
    report_dict = report.to_dict()

    if report.status == "ok" and args.review_sample:
        sample_path = report_dir / "review_sample.jsonl"
        _write_review_sample(path, sample_path, args.review_sample, seed=42)
        report_dict["manual_checks"]["DC-R12"]["sample_path"] = str(sample_path)
        _log(logging.WARNING, "manual_check_pending", run_id=run_id, rule="DC-R12", sample_path=str(sample_path))

    if args.skip_composition:
        _log(logging.WARNING, "composition_skipped", run_id=run_id, reason="--skip-composition")

    for warning in report.warnings:
        _log(logging.WARNING, warning.get("code", "warning"), run_id=run_id, **{k: v for k, v in warning.items() if k != "code"})

    for error in report.errors:
        _log(
            logging.ERROR,
            "validation_error",
            run_id=run_id,
            code=error["code"],
            line=error["line"],
            message_id=error["message_id"],
        )

    try:
        _write_report_atomic(report_dict, report_path)
    except CorpusIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, operation="write_report", detail=exc.detail)
        return 1

    _log(
        logging.INFO,
        "ingest_finished",
        run_id=run_id,
        status=report.status,
        duration_ms=report.run["duration_ms"],
        report_path=str(report_path),
    )
    print(str(report_path))

    if report.status == "ok":
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
