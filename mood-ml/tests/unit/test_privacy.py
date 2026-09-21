from pathlib import Path

import pytest

import ingest.validate as ingest_validate
from tests.conftest import make_row, write_corpus

PII_NEEDLE = "joao.silva.unico@example.com"


def test_pii_text_never_appears_in_report_or_logs(
    corpus_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [
        make_row(
            message_id="syn-msg-000001",
            text=f"Meu email e {PII_NEEDLE}, pode confirmar?",
        ),
        make_row(
            message_id="syn-msg-000002",
            role="agent",
            generated_label=None,
            sent_at="2026-01-01T10:05:00Z",
            text="Claro, confirmado.",
        ),
    ]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report_dir = corpus_root / "reports"

    ingest_validate.main(
        [str(path), "--report-dir", str(report_dir), "--skip-composition", "--log-format", "json"]
    )

    report_text = (report_dir / "syn-2026-01-01-a" / "validation_report.json").read_text(
        encoding="utf-8"
    )
    assert PII_NEEDLE not in report_text

    captured = capsys.readouterr()
    assert PII_NEEDLE not in captured.err
    assert PII_NEEDLE not in captured.out


def test_error_entries_never_carry_text(corpus_root: Path) -> None:
    """A rejected corpus with PII in the offending row must still keep `text`
    out of the report (P5) — only `line`/`message_id`/`detail` describe it."""
    rows = [make_row(text=f"Fale comigo em {PII_NEEDLE}", generated_label=9.9)]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report_dir = corpus_root / "reports"

    ingest_validate.main([str(path), "--report-dir", str(report_dir), "--skip-composition"])

    report_text = (report_dir / "syn-2026-01-01-a" / "validation_report.json").read_text(
        encoding="utf-8"
    )
    assert PII_NEEDLE not in report_text
