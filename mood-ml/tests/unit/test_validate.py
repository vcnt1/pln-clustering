import json
from pathlib import Path

import pytest

from ingest.validate import (
    CorpusPathError,
    _evaluate_composition,
    load_corpus,
    validate_corpus,
)
from tests.conftest import make_row, write_corpus


def _error_codes(report) -> set[str]:
    return {e["code"] for e in report.errors}


# ---------------------------------------------------------------------------
# F0 — usage errors (never produce a report)
# ---------------------------------------------------------------------------


def test_file_not_found_raises_usage_error(corpus_root: Path) -> None:
    missing = corpus_root / "raw" / "synthetic" / "nope.jsonl"
    with pytest.raises(CorpusPathError) as exc:
        validate_corpus(missing)
    assert exc.value.code == "IG_R01_FILE_NOT_FOUND"


def test_wrong_extension_raises_usage_error(corpus_root: Path) -> None:
    bad = corpus_root / "raw" / "synthetic" / "sample.txt"
    bad.parent.mkdir(parents=True)
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusPathError) as exc:
        validate_corpus(bad)
    assert exc.value.code == "IG_R01_NOT_READABLE"


def test_empty_file_raises_usage_error(corpus_root: Path) -> None:
    empty = corpus_root / "raw" / "synthetic" / "empty.jsonl"
    empty.parent.mkdir(parents=True)
    empty.touch()
    with pytest.raises(CorpusPathError) as exc:
        validate_corpus(empty)
    assert exc.value.code == "IG_R01_EMPTY_FILE"


def test_path_outside_synthetic_raises_usage_error(corpus_root: Path) -> None:
    outside = corpus_root / "elsewhere" / "corpus.jsonl"
    outside.parent.mkdir(parents=True)
    outside.write_text('{"a": 1}\n', encoding="utf-8")
    with pytest.raises(CorpusPathError) as exc:
        validate_corpus(outside)
    assert exc.value.code == "IG_R02_PATH_OUTSIDE_SYNTHETIC"


# ---------------------------------------------------------------------------
# F1 — encoding
# ---------------------------------------------------------------------------


def test_bom_is_rejected(corpus_root: Path) -> None:
    path = corpus_root / "raw" / "synthetic" / "syn-2026-01-01-a.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as f:
        f.write(b"\xef\xbb\xbf")
        f.write((json.dumps(make_row()) + "\n").encode("utf-8"))
    report = validate_corpus(path)
    assert report.status == "rejected"
    assert "IG_R04_ENCODING_INVALID" in _error_codes(report)


def test_invalid_utf8_is_rejected(corpus_root: Path) -> None:
    path = corpus_root / "raw" / "synthetic" / "syn-2026-01-01-a.jsonl"
    path.parent.mkdir(parents=True)
    with path.open("wb") as f:
        f.write((json.dumps(make_row()) + "\n").encode("utf-8"))
        f.write(b"\xff\xfe not valid utf-8\n")
    report = validate_corpus(path)
    assert report.status == "rejected"
    assert "IG_R04_ENCODING_INVALID" in _error_codes(report)


# ---------------------------------------------------------------------------
# F3 — schema, one case per code (CA-02)
# ---------------------------------------------------------------------------


def test_invalid_json_line(corpus_root: Path) -> None:
    path = corpus_root / "raw" / "synthetic" / "syn-2026-01-01-a.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json\n", encoding="utf-8")
    report = validate_corpus(path)
    assert _error_codes(report) == {"DC_R01_INVALID_JSON"}


@pytest.mark.parametrize(
    "overrides,mutate",
    [
        ({}, lambda r: r.pop("persona")),  # missing field
        ({}, lambda r: r.__setitem__("extra_field", "x")),  # extra field
        ({"text": "   "}, None),  # empty after strip
        ({"text": "x" * 1001}, None),  # too long
        ({"sent_at": "2026-01-01 10:00:00"}, None),  # malformed sent_at
        ({"role": "bot"}, None),  # invalid role
        ({"persona": "NotSnakeCase"}, None),  # persona not snake_case
    ],
)
def test_schema_violation_variants(corpus_root: Path, overrides: dict, mutate) -> None:
    row = make_row(**overrides)
    if mutate:
        mutate(row)
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report = validate_corpus(path)
    assert _error_codes(report) == {"DC_R01_SCHEMA_VIOLATION"}


def test_customer_label_missing_or_invalid(corpus_root: Path) -> None:
    row = make_row(role="customer", generated_label=0.3)
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report = validate_corpus(path)
    assert _error_codes(report) == {"DC_R07_LABEL_MISSING_OR_INVALID"}


def test_agent_label_not_null(corpus_root: Path) -> None:
    row = make_row(role="agent", generated_label=0.0)
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report = validate_corpus(path)
    assert _error_codes(report) == {"DC_R08_AGENT_LABEL_NOT_NULL"}


def test_id_prefix_invalid(corpus_root: Path) -> None:
    row = make_row(customer_id="cust-0001")
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report = validate_corpus(path)
    assert _error_codes(report) == {"DC_R03_ID_PREFIX_INVALID"}


def test_corpus_id_inconsistent(corpus_root: Path) -> None:
    row = make_row(corpus_id="syn-2026-02-02-b")
    path = write_corpus(corpus_root, "syn-2026-01-01-a", [row])
    report = validate_corpus(path)
    assert _error_codes(report) == {"IG_R03_CORPUS_ID_INCONSISTENT"}


# ---------------------------------------------------------------------------
# F4 — cross-line integrity
# ---------------------------------------------------------------------------


def test_duplicate_message_id(corpus_root: Path) -> None:
    rows = [
        make_row(message_id="syn-msg-000001", sent_at="2026-01-01T10:00:00Z"),
        make_row(message_id="syn-msg-000001", sent_at="2026-01-01T10:05:00Z"),
    ]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report = validate_corpus(path)
    assert "DC_R02_DUPLICATE_MESSAGE_ID" in _error_codes(report)


def test_conversation_multi_customer(corpus_root: Path) -> None:
    rows = [
        make_row(message_id="syn-msg-000001", customer_id="syn-cust-0001"),
        make_row(message_id="syn-msg-000002", customer_id="syn-cust-0002", sent_at="2026-01-01T10:05:00Z"),
    ]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report = validate_corpus(path)
    assert "DC_R04_CONVERSATION_MULTI_CUSTOMER" in _error_codes(report)


def test_customer_multi_persona(corpus_root: Path) -> None:
    rows = [
        make_row(message_id="syn-msg-000001", conversation_id="syn-conv-0001", persona="objetivo"),
        make_row(
            message_id="syn-msg-000002",
            conversation_id="syn-conv-0002",
            persona="cordial",
            sent_at="2026-01-01T10:05:00Z",
        ),
    ]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report = validate_corpus(path)
    assert "DC_R05_CUSTOMER_MULTI_PERSONA" in _error_codes(report)


def test_sent_at_duplicate_in_conversation(corpus_root: Path) -> None:
    rows = [
        make_row(message_id="syn-msg-000001", sent_at="2026-01-01T10:00:00Z"),
        make_row(message_id="syn-msg-000002", sent_at="2026-01-01T10:00:00Z"),
    ]
    path = write_corpus(corpus_root, "syn-2026-01-01-a", rows)
    report = validate_corpus(path)
    assert "DC_R06_SENT_AT_DUPLICATE" in _error_codes(report)


# ---------------------------------------------------------------------------
# F5 — composition (unit-level, via the internal aggregator)
# ---------------------------------------------------------------------------


def test_composition_label_level_underrepresented() -> None:
    label_counts = {-1.0: 0, -0.5: 0, 0.0: 100, 0.5: 0, 1.0: 0}
    _, _, errors, _ = _evaluate_composition(
        customer_message_count=100,
        conversation_ids={"c1"},
        customer_ids={"u1"},
        label_counts=label_counts,
        persona_customers={"objetivo": {"u1"}},
        conv_message_count={"c1": 4},
        conv_roles={"c1": {"customer", "agent"}},
        conv_labels={"c1": {0.0}},
        pii_counts={"cpf": 0, "phone": 0, "email": 5},
        customer_messages_with_pii=8,
        skip_composition=False,
    )
    codes = {e[0] for e in errors}
    assert "DC_R10_LABEL_LEVEL_UNDERREPRESENTED" in codes


def test_composition_persona_underrepresented() -> None:
    _, _, errors, _ = _evaluate_composition(
        customer_message_count=100,
        conversation_ids={"c1"},
        customer_ids={f"u{i}" for i in range(100)},
        label_counts={-1.0: 20, -0.5: 20, 0.0: 20, 0.5: 20, 1.0: 20},
        persona_customers={"objetivo": {f"u{i}" for i in range(97)}, "cordial": {"u97", "u98", "u99"}},
        conv_message_count={"c1": 4},
        conv_roles={"c1": {"customer", "agent"}},
        conv_labels={"c1": {0.0}},
        pii_counts={"cpf": 0, "phone": 0, "email": 7},
        customer_messages_with_pii=7,
        skip_composition=False,
    )
    codes = {e[0] for e in errors}
    assert "DC_R11_PERSONA_UNDERREPRESENTED" in codes


def test_skip_composition_exempts_only_dc_r09_to_r11() -> None:
    _, _, errors, _ = _evaluate_composition(
        customer_message_count=10,
        conversation_ids={"c1"},
        customer_ids={"u1"},
        label_counts={-1.0: 10, -0.5: 0, 0.0: 0, 0.5: 0, 1.0: 0},
        persona_customers={"objetivo": {"u1"}},
        conv_message_count={"c1": 2},
        conv_roles={"c1": {"customer"}},
        conv_labels={"c1": {-1.0}},
        pii_counts={"cpf": 0, "phone": 0, "email": 0},
        customer_messages_with_pii=0,
        skip_composition=True,
    )
    codes = {e[0] for e in errors}
    assert "DC_R09_COMPOSITION_BELOW_MINIMUM" not in codes
    assert "DC_R10_LABEL_LEVEL_UNDERREPRESENTED" not in codes
    assert "DC_R11_PERSONA_UNDERREPRESENTED" not in codes
    # not exempted:
    assert "DC_R14_PII_RATIO_OUT_OF_RANGE" in codes  # 0% PII, outside [5%,15%]
    assert "DC_R15_CONVERSATION_LENGTH_INVALID" in codes  # length 2 < 4
    assert "DC_R15_CONVERSATION_SINGLE_ROLE" in codes  # missing "agent"
    assert "DC_R16_MOOD_VARIATION_INSUFFICIENT" in codes  # 0% variation < 30%


def test_composition_passes_with_no_findings() -> None:
    _, _, errors, _ = _evaluate_composition(
        customer_message_count=10,
        conversation_ids={"c1"},
        customer_ids={"u1"},
        label_counts={-1.0: 2, -0.5: 2, 0.0: 2, 0.5: 2, 1.0: 2},
        persona_customers={"objetivo": {"u1"}},
        conv_message_count={"c1": 8},
        conv_roles={"c1": {"customer", "agent"}},
        conv_labels={"c1": {-1.0, 1.0}},
        pii_counts={"cpf": 1, "phone": 0, "email": 0},
        customer_messages_with_pii=1,
        skip_composition=True,
    )
    assert errors == []


# ---------------------------------------------------------------------------
# CA-03 — the committed sample.jsonl fixture
# ---------------------------------------------------------------------------


def test_sample_fixture_fails_composition_without_flag(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())

    report = validate_corpus(dest, skip_composition=False)
    assert report.status == "rejected"
    assert "DC_R09_COMPOSITION_BELOW_MINIMUM" in _error_codes(report)


def test_sample_fixture_passes_with_skip_composition(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())

    report = validate_corpus(dest, skip_composition=True)
    assert report.status == "ok"


# ---------------------------------------------------------------------------
# CA-08 — ordering is a warning, never a rejection; load_corpus() sorts
# ---------------------------------------------------------------------------


def test_shuffled_file_warns_but_does_not_reject(corpus_root: Path, sample_jsonl_path: Path) -> None:
    rows = [json.loads(line) for line in sample_jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    shuffled = list(reversed(rows))  # deterministic "out of order" without needing `random`
    path = write_corpus(corpus_root, "syn-2026-01-01-a", shuffled)

    report = validate_corpus(path, skip_composition=True)
    assert report.status == "ok"
    warning_codes = {w["code"] for w in report.warnings}
    assert "file_unordered" in warning_codes

    ordered = load_corpus(path)
    sent_ats_by_conv: dict[str, list[str]] = {}
    for m in ordered:
        sent_ats_by_conv.setdefault(m.conversation_id, []).append(m.sent_at)
    for values in sent_ats_by_conv.values():
        assert values == sorted(values)


# ---------------------------------------------------------------------------
# CA-10 — validate_corpus() output matches what the CLI would write
# ---------------------------------------------------------------------------


def test_validate_corpus_output_is_json_serializable(corpus_root: Path, sample_jsonl_path: Path) -> None:
    dest_dir = corpus_root / "raw" / "synthetic"
    dest_dir.mkdir(parents=True)
    dest = dest_dir / "syn-2026-01-01-a.jsonl"
    dest.write_bytes(sample_jsonl_path.read_bytes())

    report = validate_corpus(dest, skip_composition=True)
    serialized = json.dumps(report.to_dict(), sort_keys=True)
    reparsed = json.loads(serialized)
    assert reparsed["status"] == "ok"
    assert reparsed["source"]["corpus_id"] == "syn-2026-01-01-a"
