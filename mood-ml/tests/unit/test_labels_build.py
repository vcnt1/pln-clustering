import json
from pathlib import Path

import pytest

from labels.build import LabelGateError, LabelSanityError, build_labels
from tests.conftest import approved_sample_corpus, write_corpus


def _sample_rows(sample_jsonl_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in sample_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_gate_missing_report_raises(corpus_root: Path, sample_jsonl_path: Path) -> None:
    path = write_corpus(corpus_root, "syn-2026-01-01-a", _sample_rows(sample_jsonl_path))
    with pytest.raises(LabelGateError) as exc:
        build_labels(path, report_dir=corpus_root / "reports")
    assert exc.value.code == "LB_R01_GATE_MISSING_REPORT"


def test_gate_sha256_mismatch_raises(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"tampered": true}\n')  # changes the file's bytes after approval

    with pytest.raises(LabelGateError) as exc:
        build_labels(path, report_dir=corpus_root / "reports")
    assert exc.value.code == "LB_R02_GATE_SHA256_MISMATCH"


def test_build_labels_filters_customer_only(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    rows, meta = build_labels(path, report_dir=corpus_root / "reports")

    assert len(rows) == 30  # sample.jsonl has 30 customer / 26 agent messages
    assert meta["messages_customer"] == 30
    assert meta["messages_agent_excluded"] == 26


def test_label_row_schema_has_exactly_the_spec_columns(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    rows, _ = build_labels(path, report_dir=corpus_root / "reports")

    expected_columns = {
        "label_set_id",
        "target_type",
        "target_id",
        "label_score",
        "scale",
        "label_source",
        "annotator",
        "labeled_at",
    }
    assert set(rows[0].keys()) == expected_columns
    assert "persona" not in rows[0]
    assert "customer_id" not in rows[0]
    assert "text" not in rows[0]


def test_label_set_id_is_deterministic_from_corpus_id(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    rows, meta = build_labels(path, report_dir=corpus_root / "reports")
    assert meta["label_set_id"] == "ls-syn-2026-01-01-a"
    assert all(r["label_set_id"] == "ls-syn-2026-01-01-a" for r in rows)


def test_labeled_at_comes_from_message_sent_at(corpus_root: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    rows, _ = build_labels(path, report_dir=corpus_root / "reports")
    first = next(r for r in rows if r["target_id"] == "syn-msg-000001")
    assert first["labeled_at"] == "2026-08-24T11:41:34Z"
    assert first["label_score"] == 0.0


def test_rows_preserve_load_corpus_order(corpus_root: Path) -> None:
    from ingest.validate import load_corpus

    path = approved_sample_corpus(corpus_root)
    rows, _ = build_labels(path, report_dir=corpus_root / "reports")
    expected_order = [m.message_id for m in load_corpus(path) if m.role == "customer"]
    assert [r["target_id"] for r in rows] == expected_order


def test_sanity_check_catches_a_broken_row_filter(
    corpus_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CA-05: a deliberately broken row-building step must be caught by the
    independently-computed sanity check, LB_R10_ROW_COUNT_MISMATCH."""
    import labels.build as labels_build

    path = approved_sample_corpus(corpus_root)
    monkeypatch.setattr(labels_build, "_build_label_rows", lambda msgs, lsid: [])

    with pytest.raises(LabelSanityError) as exc:
        build_labels(path, report_dir=corpus_root / "reports")
    assert exc.value.code == "LB_R10_ROW_COUNT_MISMATCH"


def test_sanity_check_catches_label_score_out_of_domain(
    corpus_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import labels.build as labels_build

    path = approved_sample_corpus(corpus_root)
    real_build_rows = labels_build._build_label_rows

    def _corrupt_scores(msgs, lsid):
        rows = real_build_rows(msgs, lsid)
        rows[0]["label_score"] = 9.9
        return rows

    monkeypatch.setattr(labels_build, "_build_label_rows", _corrupt_scores)

    with pytest.raises(LabelSanityError) as exc:
        build_labels(path, report_dir=corpus_root / "reports")
    assert exc.value.code == "LB_R11_LABEL_SCORE_OUT_OF_DOMAIN"


def test_sanity_check_catches_duplicate_target_id(
    corpus_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import labels.build as labels_build

    path = approved_sample_corpus(corpus_root)
    real_build_rows = labels_build._build_label_rows

    def _corrupt_ids(msgs, lsid):
        rows = real_build_rows(msgs, lsid)
        rows[1]["target_id"] = rows[0]["target_id"]
        return rows

    monkeypatch.setattr(labels_build, "_build_label_rows", _corrupt_ids)

    with pytest.raises(LabelSanityError) as exc:
        build_labels(path, report_dir=corpus_root / "reports")
    assert exc.value.code == "LB_R11_DUPLICATE_TARGET_ID"


def test_no_text_anywhere_in_output(corpus_root: Path, sample_jsonl_path: Path) -> None:
    path = approved_sample_corpus(corpus_root)
    rows, meta = build_labels(path, report_dir=corpus_root / "reports")

    corpus_texts = {row["text"] for row in _sample_rows(sample_jsonl_path)}
    output_blob = str(rows) + str(meta)
    for text in corpus_texts:
        assert text not in output_blob
