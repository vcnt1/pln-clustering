import json
from pathlib import Path

import labels.build as labels_build
from tests.conftest import approved_sample_corpus, write_corpus


def _sample_rows(sample_jsonl_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in sample_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_exit_0_on_approved_corpus(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    labels_dir = corpus_root / "labels"
    report_dir = corpus_root / "reports"

    code = labels_build.main(
        [str(path), "--labels-dir", str(labels_dir), "--report-dir", str(report_dir)]
    )
    assert code == 0
    assert (labels_dir / "ls-syn-2026-01-01-a.parquet").exists()
    report = json.loads((report_dir / "syn-2026-01-01-a" / "label_report.json").read_text(encoding="utf-8"))
    assert report["label_set"]["rows"] == 30


def test_exit_3_when_corpus_not_approved(corpus_root: Path, sample_jsonl_path: Path) -> None:
    path = write_corpus(corpus_root, "syn-2026-01-01-a", _sample_rows(sample_jsonl_path))
    labels_dir = corpus_root / "labels"
    report_dir = corpus_root / "reports"

    code = labels_build.main(
        [str(path), "--labels-dir", str(labels_dir), "--report-dir", str(report_dir)]
    )
    assert code == 3
    assert not labels_dir.exists()


def test_overwrite_is_byte_identical_across_runs(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    labels_dir = corpus_root / "labels"
    report_dir = corpus_root / "reports"

    code1 = labels_build.main([str(path), "--labels-dir", str(labels_dir), "--report-dir", str(report_dir)])
    assert code1 == 0
    parquet_bytes_1 = (labels_dir / "ls-syn-2026-01-01-a.parquet").read_bytes()

    code2 = labels_build.main([str(path), "--labels-dir", str(labels_dir), "--report-dir", str(report_dir)])
    assert code2 == 0
    parquet_bytes_2 = (labels_dir / "ls-syn-2026-01-01-a.parquet").read_bytes()

    assert parquet_bytes_1 == parquet_bytes_2


def test_messages_customer_matches_ingest_report(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    labels_dir = corpus_root / "labels"
    report_dir = corpus_root / "reports"
    labels_build.main([str(path), "--labels-dir", str(labels_dir), "--report-dir", str(report_dir)])

    ingest_report = json.loads((report_dir / "syn-2026-01-01-a" / "validation_report.json").read_text(encoding="utf-8"))
    label_report = json.loads((report_dir / "syn-2026-01-01-a" / "label_report.json").read_text(encoding="utf-8"))
    assert label_report["counts"]["messages_customer"] == ingest_report["counts"]["messages_customer"]
