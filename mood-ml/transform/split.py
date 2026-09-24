"""[4] Splits the labeled corpus into train/validation/test grouped by customer_id (see specs/05-split.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from common.errors import PipelineError
from common.io import atomic_write, with_io_retry, write_json
from common.log import configure_logging, log, now_iso
from ingest.validate import (
    IngestGateError,
    check_ingest_gate,
    load_corpus,
    sha256_and_size,
)
from transform.features import FEATURE_SPEC_VERSION, extract_features

DEFAULT_SPLIT_CONFIG = {"train_ratio": 0.70, "validation_ratio": 0.15, "test_ratio": 0.15, "seed": 42}
MAX_HISTORY_SIZE = 30

logger = logging.getLogger("transform.split")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SplitGateError(PipelineError):
    pass


class DatasetIdentityConflictError(PipelineError):
    pass


class SplitSanityError(PipelineError):
    pass


class SplitIOError(PipelineError):
    pass


class _LabelsGateError(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


_log = partial(log, logger)
_configure_logging = partial(configure_logging, logger)
_with_io_retry = partial(with_io_retry, logger=logger, error=lambda detail: SplitIOError("SP_R16_IO_FAILED", detail))


# ---------------------------------------------------------------------------
# check_labels_gate (SP-R02) — local to this module, only split.py depends on
# label_report.json as an upstream artifact.
# ---------------------------------------------------------------------------


def _check_labels_gate(corpus_path: Path, report_dir: Path) -> dict[str, Any]:
    corpus_id = corpus_path.stem
    report_path = report_dir / corpus_id / "label_report.json"
    if not report_path.exists():
        raise _LabelsGateError("missing_report", f"no label_report.json for {corpus_id}")

    report = json.loads(report_path.read_text(encoding="utf-8"))

    corpus_sha256, _ = sha256_and_size(corpus_path)
    if report["source"]["corpus_sha256"] != corpus_sha256:
        raise _LabelsGateError("corpus_sha256_mismatch", "label_report.json describes a different corpus")

    label_set_path = Path(report["label_set"]["path"])
    if not label_set_path.exists():
        raise _LabelsGateError("label_set_missing", f"label set file not found: {label_set_path}")

    label_set_sha256, _ = sha256_and_size(label_set_path)
    if report["label_set"]["sha256"] != label_set_sha256:
        raise _LabelsGateError("label_set_sha256_mismatch", "label set file changed after labels.build")

    return report


# ---------------------------------------------------------------------------
# compute_fingerprint (F2, SP-R04/R05)
# ---------------------------------------------------------------------------


def compute_fingerprint(
    corpus_sha256: str, label_set_sha256: str, split_config: dict[str, Any], feature_spec_version: str
) -> str:
    payload = {
        "corpus_sha256": corpus_sha256,
        "label_set_sha256": label_set_sha256,
        "split_strategy": {
            "train_ratio": split_config["train_ratio"],
            "validation_ratio": split_config["validation_ratio"],
            "test_ratio": split_config["test_ratio"],
            "seed": split_config["seed"],
        },
        "feature_spec_version": feature_spec_version,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# build_dataset — F0+F1 (double gate) -> F2 (identity) -> F3 (load) ->
# F4 (sliding window) -> F5 (assembly) -> F6 (sanity) -> F7 (split)
# ---------------------------------------------------------------------------


def _build_history_windows(messages: list) -> dict[str, list[dict[str, str]]]:
    """F4: sliding window of up to 30 `customer` messages per conversation
    (ADR-0007), in the order `load_corpus()` already guarantees. Returns
    message_id -> history (the `extract_features`-ready window for that
    message as trigger). A separate, named function so SP-R20's contract
    test can exercise the window rule in isolation, without needing a corpus
    that would itself violate DC-R15's 20-message conversation cap."""
    buffers: dict[str, deque[dict[str, str]]] = {}
    windows: dict[str, list[dict[str, str]]] = {}
    for m in messages:
        if m.role != "customer":
            continue
        buf = buffers.setdefault(m.conversation_id, deque(maxlen=MAX_HISTORY_SIZE))
        buf.append({"message_id": m.message_id, "role": m.role, "text": m.text, "sent_at": m.sent_at})
        windows[m.message_id] = list(buf)
    return windows


def _assemble_examples(
    messages: list, windows: dict[str, list[dict[str, str]]], label_by_target_id: dict[str, float]
) -> list[dict[str, Any]]:
    """F5: one example row per `customer` message, via extract_features (never
    reimplemented). A named function so the F6 sanity check below is a real
    regression guard, testable by monkeypatching this step in isolation."""
    examples: list[dict[str, Any]] = []
    for m in messages:
        if m.role != "customer":
            continue
        features = extract_features(windows[m.message_id])
        examples.append(
            {
                "example_id": m.message_id,
                "customer_id": m.customer_id,
                "conversation_id": m.conversation_id,
                "persona": m.persona,
                "text_clean": features["text_clean"],
                "context_clean": features["context_clean"],
                "label_score": label_by_target_id.get(m.message_id),
            }
        )
    return examples


@dataclass
class DatasetResult:
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    test_df: pd.DataFrame
    dataset_json: dict[str, Any]


def build_dataset(
    corpus_path: str | Path,
    dataset_id: str,
    split_config: dict[str, Any] | None = None,
    report_dir: str | Path = Path("data/reports"),
    datasets_dir: str | Path = Path("data/datasets"),
) -> DatasetResult:
    """Returns a DatasetResult. Writes nothing; does not decide an exit code
    (same separation as validate_corpus/build_labels, spec 05 §2.2)."""
    corpus_path = Path(corpus_path)
    corpus_id = corpus_path.stem
    report_dir = Path(report_dir)
    datasets_dir = Path(datasets_dir)
    split_config = {**DEFAULT_SPLIT_CONFIG, **(split_config or {})}

    # F1 — portão duplo
    try:
        ingest_report = check_ingest_gate(corpus_path, report_dir)
    except IngestGateError as exc:
        raise SplitGateError("SP_R01_GATE_CORPUS_NOT_OK", exc.detail) from exc

    try:
        labels_report = _check_labels_gate(corpus_path, report_dir)
    except _LabelsGateError as exc:
        raise SplitGateError("SP_R02_GATE_LABELS_NOT_OK", exc.detail) from exc

    corpus_sha256 = ingest_report["source"]["sha256"]
    label_set_sha256 = labels_report["label_set"]["sha256"]
    label_set_id = labels_report["label_set"]["label_set_id"]

    # F2 — identidade do dataset_id
    fingerprint = compute_fingerprint(corpus_sha256, label_set_sha256, split_config, FEATURE_SPEC_VERSION)
    dataset_json_path = datasets_dir / dataset_id / "dataset.json"
    if dataset_json_path.exists():
        existing = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise DatasetIdentityConflictError(
                "SP_R04_DATASET_ID_CONFLICT",
                f"dataset_id {dataset_id!r} already exists with a different fingerprint",
            )
        # fingerprint igual: segue e reconstrói/sobrescreve de forma determinística.

    # F3 — carga
    messages = load_corpus(corpus_path)
    label_set_path = Path(labels_report["label_set"]["path"])
    labels_df = pd.read_parquet(label_set_path)
    label_by_target_id = dict(zip(labels_df["target_id"], labels_df["label_score"], strict=True))

    # F4 — janela deslizante por conversa
    windows = _build_history_windows(messages)

    # F5 — montagem do exemplo
    examples = _assemble_examples(messages, windows, label_by_target_id)

    # F6 — sanity (contagem)
    if len(examples) != len(labels_df):
        raise SplitSanityError(
            "SP_R11_EXAMPLE_COUNT_MISMATCH", f"{len(examples)} examples != {len(labels_df)} label rows"
        )

    # F7 — GroupShuffleSplit em duas etapas
    df = pd.DataFrame(examples)
    groups = df["customer_id"].to_numpy()
    seed = split_config["seed"]
    val_test_ratio = split_config["validation_ratio"] + split_config["test_ratio"]

    stage1 = GroupShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=seed)
    train_idx, temp_idx = next(stage1.split(df, groups=groups))

    temp_df = df.iloc[temp_idx].reset_index(drop=True)
    temp_groups = groups[temp_idx]
    test_ratio_within_temp = split_config["test_ratio"] / val_test_ratio
    stage2 = GroupShuffleSplit(n_splits=1, test_size=test_ratio_within_temp, random_state=seed)
    val_idx_rel, test_idx_rel = next(stage2.split(temp_df, groups=temp_groups))

    train_df = df.iloc[train_idx].assign(split="train").reset_index(drop=True)
    validation_df = temp_df.iloc[val_idx_rel].assign(split="validation").reset_index(drop=True)
    test_df = temp_df.iloc[test_idx_rel].assign(split="test").reset_index(drop=True)

    # F6 — sanity (vazamento de customer_id e soma)
    train_custs = set(train_df["customer_id"])
    validation_custs = set(validation_df["customer_id"])
    test_custs = set(test_df["customer_id"])
    if (train_custs & validation_custs) or (train_custs & test_custs) or (validation_custs & test_custs):
        raise SplitSanityError("SP_R12_CUSTOMER_ID_LEAK", "a customer_id appears in more than one split")

    total_rows = len(train_df) + len(validation_df) + len(test_df)
    if total_rows != len(examples):
        raise SplitSanityError(
            "SP_R13_SPLIT_ROW_SUM_MISMATCH", f"split rows sum to {total_rows}, expected {len(examples)}"
        )

    dataset_json = {
        "dataset_id": dataset_id,
        "source_ids": {"corpus_id": corpus_id},
        "label_set_id": label_set_id,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "split_strategy": {
            "algorithm": "GroupShuffleSplit",
            "train_ratio": split_config["train_ratio"],
            "validation_ratio": split_config["validation_ratio"],
            "test_ratio": split_config["test_ratio"],
            "seed": seed,
        },
        "row_counts": {
            "train": {"rows": len(train_df), "customers": len(train_custs)},
            "validation": {"rows": len(validation_df), "customers": len(validation_custs)},
            "test": {"rows": len(test_df), "customers": len(test_custs)},
        },
        "fingerprint": fingerprint,
    }

    return DatasetResult(train_df, validation_df, test_df, dataset_json)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _write_parquet_atomic(df: pd.DataFrame, dest: Path, run_id: str) -> None:
    _with_io_retry(f"write_{dest.stem}", lambda: atomic_write(dest, lambda tmp: df.to_parquet(tmp, index=False)), run_id)


def _write_json_atomic(payload: dict[str, Any], dest: Path, run_id: str) -> None:
    _with_io_retry("write_dataset_json", lambda: atomic_write(dest, lambda tmp: write_json(payload, tmp)), run_id)


def _load_split_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return dict(DEFAULT_SPLIT_CONFIG)
    with config_path.open("r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}
    return {**DEFAULT_SPLIT_CONFIG, **full_config.get("split", {})}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m transform.split")
    parser.add_argument("path", help="path to the approved corpus .jsonl file")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--datasets-dir", default="data/datasets")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    corpus_path = Path(args.path)
    split_config = _load_split_config(Path(args.config) if args.config else None)

    _log(logging.INFO, "split_started", run_id=run_id, corpus_path=str(corpus_path), dataset_id=args.dataset_id)

    try:
        result = build_dataset(
            corpus_path,
            args.dataset_id,
            split_config=split_config,
            report_dir=args.report_dir,
            datasets_dir=args.datasets_dir,
        )
    except SplitGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except DatasetIdentityConflictError as exc:
        _log(logging.ERROR, "dataset_id_conflict", run_id=run_id, detail=exc.detail)
        return 4
    except SplitSanityError as exc:
        _log(logging.ERROR, "sanity_check_failed", run_id=run_id, check=exc.code, detail=exc.detail)
        return 5
    except SplitIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    _log(
        logging.INFO,
        "corpus_and_labels_loaded",
        run_id=run_id,
        examples=len(result.train_df) + len(result.validation_df) + len(result.test_df),
    )
    _log(
        logging.INFO,
        "sanity_check_passed",
        run_id=run_id,
        checks=["row_count", "customer_id_no_overlap", "split_sum"],
    )
    _log(
        logging.INFO,
        "split_computed",
        run_id=run_id,
        train_rows=len(result.train_df),
        validation_rows=len(result.validation_df),
        test_rows=len(result.test_df),
    )

    datasets_dir = Path(args.datasets_dir) / args.dataset_id
    try:
        _write_parquet_atomic(result.train_df, datasets_dir / "train.parquet", run_id)
        _write_parquet_atomic(result.validation_df, datasets_dir / "validation.parquet", run_id)
        _write_parquet_atomic(result.test_df, datasets_dir / "test.parquet", run_id)

        dataset_json = {**result.dataset_json, "created_at": now_iso()}
        _write_json_atomic(dataset_json, datasets_dir / "dataset.json", run_id)
    except SplitIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    finished_at = datetime.now(timezone.utc)
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)
    _log(
        logging.INFO,
        "split_finished",
        run_id=run_id,
        dataset_id=args.dataset_id,
        duration_ms=duration_ms,
        dataset_json_path=str(datasets_dir / "dataset.json"),
    )
    print(str(datasets_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
