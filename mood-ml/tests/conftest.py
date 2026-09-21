import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

BASE_ROW = {
    "corpus_id": "syn-2026-01-01-a",
    "conversation_id": "syn-conv-0001",
    "customer_id": "syn-cust-0001",
    "message_id": "syn-msg-000001",
    "role": "customer",
    "text": "Qual o horario do check-out?",
    "sent_at": "2026-01-01T10:00:00Z",
    "persona": "objetivo",
    "generated_label": 0.0,
}


def make_row(**overrides: object) -> dict:
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def write_corpus(directory: Path, corpus_id: str, rows: list[dict]) -> Path:
    """Writes `rows` as a .jsonl corpus under <directory>/raw/synthetic/<corpus_id>.jsonl,
    satisfying IG-R02 (path must be under data/raw/synthetic/)."""
    target_dir = directory / "raw" / "synthetic"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{corpus_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def approve_corpus(directory: Path, corpus_id: str, rows: list[dict], *, skip_composition: bool = True) -> Path:
    """Writes a corpus and runs the real validate_corpus() over it, so
    downstream tests (labels, split) start from a genuine, schema-correct
    validation_report.json instead of a hand-forged one. Returns the corpus
    path. Raises if the corpus is not approved (status != "ok")."""
    from ingest.validate import _write_report_atomic, validate_corpus

    path = write_corpus(directory, corpus_id, rows)
    report = validate_corpus(path, skip_composition=skip_composition)
    if report.status != "ok":
        raise AssertionError(f"approve_corpus: corpus was rejected: {report.errors}")
    report_path = directory / "reports" / corpus_id / "validation_report.json"
    _write_report_atomic(report.to_dict(), report_path)
    return path


def approved_sample_corpus(directory: Path, corpus_id: str = "syn-2026-01-01-a") -> Path:
    """Copies the committed sample.jsonl fixture (56 lines, 6 real conversations,
    already known to satisfy DC-R14/R15/R16) into <directory>/raw/synthetic/ and
    runs the real validator, producing a genuine validation_report.json. Used by
    labels/split tests that need a realistic corpus without composition minimums
    (--skip-composition covers only DC-R09..R11, not R14/R15/R16 — a hand-rolled
    1-2 message corpus fails those and can't be used here)."""
    sample_path = Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["corpus_id"] = corpus_id
    return approve_corpus(directory, corpus_id, rows, skip_composition=True)


def approved_medium_corpus(directory: Path, corpus_id: str = "syn-2026-01-02-a") -> Path:
    """Like approved_sample_corpus, but backed by tests/fixtures/sample_medium.jsonl
    (40 customers, 463 lines) — split.py's GroupShuffleSplit needs enough groups
    for train/validation/test to all come out non-empty, which the 6-customer
    sample.jsonl is too small to guarantee."""
    medium_path = Path(__file__).resolve().parent / "fixtures" / "sample_medium.jsonl"
    rows = [json.loads(line) for line in medium_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["corpus_id"] = corpus_id
    return approve_corpus(directory, corpus_id, rows, skip_composition=True)


@pytest.fixture
def medium_jsonl_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sample_medium.jsonl"


def approve_labels(directory: Path, corpus_path: Path) -> Path:
    """Runs the real build_labels() + writes the label set and label_report.json,
    so split tests start from a genuine label gate. Returns the .parquet path."""
    from ingest.validate import sha256_and_size
    from labels.build import _write_parquet_atomic, build_labels
    from labels.build import _write_report_atomic as _write_label_report

    report_dir = directory / "reports"
    labels_dir = directory / "labels"
    rows, meta = build_labels(corpus_path, report_dir)

    parquet_path = labels_dir / f"{meta['label_set_id']}.parquet"
    _write_parquet_atomic(rows, parquet_path, run_id="test")
    parquet_sha256, _ = sha256_and_size(parquet_path)

    report = {
        "report_version": "lb-1",
        "run": {"run_id": "test", "tool": "labels.build"},
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
        "distributions": {"label_levels": {}},
    }
    report_path = report_dir / meta["corpus_id"] / "label_report.json"
    _write_label_report(report, report_path, run_id="test")
    return parquet_path


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_jsonl_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
