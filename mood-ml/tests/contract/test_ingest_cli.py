import json
from pathlib import Path

import pytest

import ingest.validate as ingest_validate
from tests.conftest import make_row, write_corpus


def _report_path(report_dir: Path, corpus_id: str) -> Path:
    return report_dir / corpus_id / "validation_report.json"


def test_exit_0_on_valid_corpus(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())
    report_dir = corpus_root / "reports"

    code = ingest_validate.main(
        [str(dest), "--report-dir", str(report_dir), "--skip-composition"]
    )
    assert code == 0
    report = json.loads(_report_path(report_dir, "syn-2026-01-01-a").read_text(encoding="utf-8"))
    assert report["status"] == "ok"


def test_exit_2_on_usage_error_writes_no_report(corpus_root: Path) -> None:
    missing = corpus_root / "raw" / "synthetic" / "nope.jsonl"
    report_dir = corpus_root / "reports"

    code = ingest_validate.main([str(missing), "--report-dir", str(report_dir)])
    assert code == 2
    assert not report_dir.exists()


def test_exit_3_on_rejected_corpus(corpus_root: Path) -> None:
    row = make_row(role="agent", generated_label=0.0)  # DC_R08 violation
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report_dir = corpus_root / "reports"

    code = ingest_validate.main([str(path), "--report-dir", str(report_dir)])
    assert code == 3
    report = json.loads(_report_path(report_dir, "syn-2026-01-01-a").read_text(encoding="utf-8"))
    assert report["status"] == "rejected"


def test_idempotent_rerun_does_not_rewrite_report(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())
    report_dir = corpus_root / "reports"

    code1 = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code1 == 0
    mtime_1 = _report_path(report_dir, "syn-2026-01-01-a").stat().st_mtime_ns

    code2 = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code2 == 0
    mtime_2 = _report_path(report_dir, "syn-2026-01-01-a").stat().st_mtime_ns

    assert mtime_1 == mtime_2


def test_hash_conflict_returns_exit_4_and_preserves_report(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())
    report_dir = corpus_root / "reports"

    code1 = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code1 == 0
    original_report = _report_path(report_dir, "syn-2026-01-01-a").read_text(encoding="utf-8")

    # Same corpus_id, different bytes.
    extra_row = make_row(message_id="syn-msg-999999", sent_at="2026-01-01T23:59:00Z")
    with dest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(extra_row, ensure_ascii=False) + "\n")

    code2 = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code2 == 4
    assert _report_path(report_dir, "syn-2026-01-01-a").read_text(encoding="utf-8") == original_report


def test_io_failure_after_retries_returns_exit_1(
    corpus_root: Path, sample_jsonl_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())
    report_dir = corpus_root / "reports"

    monkeypatch.setattr(ingest_validate.time, "sleep", lambda _seconds: None)

    def _always_fails(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated permission error")

    monkeypatch.setattr(ingest_validate.os, "replace", _always_fails)

    code = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code == 1


def test_io_retry_succeeds_after_transient_failures(
    corpus_root: Path, sample_jsonl_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())
    report_dir = corpus_root / "reports"

    monkeypatch.setattr(ingest_validate.time, "sleep", lambda _seconds: None)

    real_replace = ingest_validate.os.replace
    attempts = {"n": 0}

    def _flaky_replace(*args: object, **kwargs: object) -> None:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise OSError("simulated transient lock")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(ingest_validate.os, "replace", _flaky_replace)

    code = ingest_validate.main([str(dest), "--report-dir", str(report_dir), "--skip-composition"])
    assert code == 0
    assert attempts["n"] == 2
