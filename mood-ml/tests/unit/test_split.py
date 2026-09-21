from pathlib import Path

import pytest

from ingest.validate import Message
from tests.conftest import approve_labels, approved_medium_corpus, write_corpus
from transform.split import (
    DatasetIdentityConflictError,
    SplitGateError,
    SplitSanityError,
    _build_history_windows,
    build_dataset,
    compute_fingerprint,
)

# ---------------------------------------------------------------------------
# SP-R20 — contract test: the sliding window vs. an independent oracle
# ---------------------------------------------------------------------------


def _msg(i: int, role: str, conv: str = "syn-conv-0001") -> Message:
    minute = i % 60
    hour = i // 60
    return Message(
        corpus_id="syn-2026-01-01-a",
        conversation_id=conv,
        customer_id="syn-cust-0001",
        message_id=f"syn-msg-{i:06d}",
        role=role,
        text=f"mensagem numero {i}",
        sent_at=f"2026-01-01T{hour:02d}:{minute:02d}:00Z",
        persona="objetivo",
        generated_label=0.0 if role == "customer" else None,
    )


def _oracle_history(messages: list[Message], target_message_id: str) -> list[dict[str, str]]:
    """Independent (naive, O(n^2)) reimplementation of the ADR-0007 rule,
    written only for this test — deliberately not reusing _build_history_windows,
    so the comparison proves the production window matches the *rule*, not
    just itself."""
    target = next(m for m in messages if m.message_id == target_message_id)
    same_conv_customer = [
        m
        for m in messages
        if m.conversation_id == target.conversation_id and m.role == "customer" and m.sent_at <= target.sent_at
    ]
    same_conv_customer.sort(key=lambda m: m.sent_at)
    window = same_conv_customer[-30:]
    return [{"message_id": m.message_id, "role": m.role, "text": m.text, "sent_at": m.sent_at} for m in window]


def test_sliding_window_matches_independent_oracle_over_45_messages() -> None:
    messages: list[Message] = []
    for i in range(45):
        messages.append(_msg(i * 2, "customer"))
        messages.append(_msg(i * 2 + 1, "agent"))
    messages.sort(key=lambda m: m.sent_at)

    windows = _build_history_windows(messages)
    customer_ids = [m.message_id for m in messages if m.role == "customer"]

    for target_id in customer_ids:
        assert windows[target_id] == _oracle_history(messages, target_id)

    last_customer_id = customer_ids[-1]
    assert len(windows[last_customer_id]) == 30
    assert all(item["role"] == "customer" for item in windows[last_customer_id])
    assert windows[last_customer_id][-1]["message_id"] == last_customer_id
    assert windows[last_customer_id] == sorted(windows[last_customer_id], key=lambda item: item["sent_at"])


def test_sliding_window_excludes_agent_and_other_conversations() -> None:
    messages = [
        _msg(0, "customer", conv="syn-conv-0001"),
        _msg(1, "agent", conv="syn-conv-0001"),
        _msg(2, "customer", conv="syn-conv-0002"),  # different conversation
        _msg(3, "customer", conv="syn-conv-0001"),
    ]
    windows = _build_history_windows(messages)
    last_id = "syn-msg-000003"
    assert [item["message_id"] for item in windows[last_id]] == ["syn-msg-000000", "syn-msg-000003"]


def test_single_message_conversation_has_window_of_one() -> None:
    messages = [_msg(0, "customer")]
    windows = _build_history_windows(messages)
    assert len(windows["syn-msg-000000"]) == 1


# ---------------------------------------------------------------------------
# Gate (SP-R01/R02) and identity (SP-R04)
# ---------------------------------------------------------------------------


def test_gate_blocked_when_corpus_not_approved(corpus_root: Path, medium_jsonl_path: Path) -> None:
    import json

    rows = [json.loads(line) for line in medium_jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    path = write_corpus(corpus_root, "syn-2026-01-02-a", rows)

    with pytest.raises(SplitGateError) as exc:
        build_dataset(path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets")
    assert exc.value.code == "SP_R01_GATE_CORPUS_NOT_OK"


def test_gate_blocked_when_labels_not_built(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)

    with pytest.raises(SplitGateError) as exc:
        build_dataset(path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets")
    assert exc.value.code == "SP_R02_GATE_LABELS_NOT_OK"


def test_dataset_id_conflict_when_fingerprint_differs(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)

    result1 = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )
    from transform.split import _write_json_atomic

    dataset_dir = corpus_root / "datasets" / "ds-2026-01-02-a"
    _write_json_atomic(result1.dataset_json, dataset_dir / "dataset.json", run_id="test")

    with pytest.raises(DatasetIdentityConflictError):
        build_dataset(
            path,
            "ds-2026-01-02-a",
            split_config={"train_ratio": 0.5, "validation_ratio": 0.25, "test_ratio": 0.25, "seed": 42},
            report_dir=corpus_root / "reports",
            datasets_dir=corpus_root / "datasets",
        )


def test_dataset_id_reused_when_fingerprint_matches(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)

    result1 = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )
    from transform.split import _write_json_atomic

    dataset_dir = corpus_root / "datasets" / "ds-2026-01-02-a"
    _write_json_atomic(result1.dataset_json, dataset_dir / "dataset.json", run_id="test")

    result2 = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )
    assert result1.dataset_json["fingerprint"] == result2.dataset_json["fingerprint"]
    assert result1.train_df["example_id"].tolist() == result2.train_df["example_id"].tolist()


# ---------------------------------------------------------------------------
# Split correctness
# ---------------------------------------------------------------------------


def test_no_customer_id_overlap_between_splits(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)
    result = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )

    train_custs = set(result.train_df["customer_id"])
    validation_custs = set(result.validation_df["customer_id"])
    test_custs = set(result.test_df["customer_id"])

    assert not (train_custs & validation_custs)
    assert not (train_custs & test_custs)
    assert not (validation_custs & test_custs)
    assert len(train_custs) > 0
    assert len(validation_custs) > 0
    assert len(test_custs) > 0


def test_row_counts_match_dataset_json(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)
    result = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )

    counts = result.dataset_json["row_counts"]
    assert counts["train"]["rows"] == len(result.train_df)
    assert counts["validation"]["rows"] == len(result.validation_df)
    assert counts["test"]["rows"] == len(result.test_df)
    total = len(result.train_df) + len(result.validation_df) + len(result.test_df)
    assert total == counts["train"]["rows"] + counts["validation"]["rows"] + counts["test"]["rows"]


def test_example_schema_has_exactly_the_spec_columns(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)
    result = build_dataset(
        path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets"
    )
    expected_columns = {
        "example_id",
        "customer_id",
        "conversation_id",
        "persona",
        "text_clean",
        "context_clean",
        "label_score",
        "split",
    }
    assert set(result.train_df.columns) == expected_columns


def test_sanity_check_catches_example_count_mismatch(
    corpus_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import transform.split as split_mod

    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)

    monkeypatch.setattr(split_mod, "_assemble_examples", lambda messages, windows, labels: [])

    with pytest.raises(SplitSanityError) as exc:
        build_dataset(path, "ds-2026-01-02-a", report_dir=corpus_root / "reports", datasets_dir=corpus_root / "datasets")
    assert exc.value.code == "SP_R11_EXAMPLE_COUNT_MISMATCH"


def test_compute_fingerprint_is_deterministic_and_sensitive_to_config() -> None:
    config_a = {"train_ratio": 0.7, "validation_ratio": 0.15, "test_ratio": 0.15, "seed": 42}
    config_b = {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2, "seed": 42}

    fp1 = compute_fingerprint("abc", "def", config_a, "fs-1")
    fp2 = compute_fingerprint("abc", "def", config_a, "fs-1")
    fp3 = compute_fingerprint("abc", "def", config_b, "fs-1")

    assert fp1 == fp2
    assert fp1 != fp3
