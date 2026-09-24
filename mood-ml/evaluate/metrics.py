"""[6] Computes test metrics and applies the quality gate (see specs/06-train-evaluate.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from common.errors import PipelineError
from common.io import atomic_write, with_io_retry, write_json
from common.log import configure_logging, log, now_iso
from transform.features import FEATURE_SPEC_VERSION

logger = logging.getLogger("evaluate.metrics")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EvaluateGateError(PipelineError):
    pass


class EvaluateIOError(PipelineError):
    pass


_log = partial(log, logger)
_configure_logging = partial(configure_logging, logger)
_with_io_retry = partial(with_io_retry, logger=logger, error=lambda detail: EvaluateIOError("EV_IO_FAILED", detail))


# ---------------------------------------------------------------------------
# clip_score — shared with train.train and (later) infer/predict.py, spec 08
# ---------------------------------------------------------------------------


def clip_score(raw: float) -> float:
    """Clips a raw regression output to [-1.0, 1.0] (ADR-0001)."""
    return max(-1.0, min(1.0, float(raw)))


# ---------------------------------------------------------------------------
# compute_training_fingerprint — shared with train.train. Lives here (not in
# train/train.py) so train.py can import clip_score + this from one module
# without a circular import (train.py -> evaluate.metrics, never the reverse).
# ---------------------------------------------------------------------------


def compute_training_fingerprint(
    dataset_sha256: dict[str, str],
    algorithm: str,
    hyperparameters: dict[str, Any],
    feature_spec_version: str,
) -> str:
    payload = {
        "dataset_sha256": dataset_sha256,
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "feature_spec_version": feature_spec_version,
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def dataset_parquet_hashes(datasets_dir: Path, dataset_id: str) -> dict[str, str]:
    from ingest.validate import sha256_and_size

    base = Path(datasets_dir) / dataset_id
    return {split: sha256_and_size(base / f"{split}.parquet")[0] for split in ("train", "validation", "test")}


def staging_path(staging_dir: Path, dataset_id: str, algorithm: str, fingerprint: str) -> Path:
    return Path(staging_dir) / dataset_id / algorithm / fingerprint


# ---------------------------------------------------------------------------
# evaluate_candidate — F1 (gate) -> F2 (load) -> F3 (metrics) -> F4 (gate)
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    eval_json: dict[str, Any]
    gate_passed: bool
    staging_dir: Path
    algorithm: str


def evaluate_candidate(
    dataset_id: str,
    config: dict[str, Any],
    fingerprint: str | None = None,
    staging_dir: str | Path = Path("models/_staging"),
    datasets_dir: str | Path = Path("data/datasets"),
) -> EvalResult:
    """Returns an EvalResult. Writes nothing; does not decide an exit code
    (same separation as validate_corpus/build_labels/build_dataset)."""
    staging_dir = Path(staging_dir)
    datasets_dir = Path(datasets_dir)
    train_config = dict(config.get("train", {}))
    algorithm = train_config.pop("algorithm", "tfidf-ridge")

    if fingerprint is None:
        dataset_sha256 = dataset_parquet_hashes(datasets_dir, dataset_id)
        fingerprint = compute_training_fingerprint(dataset_sha256, algorithm, train_config, FEATURE_SPEC_VERSION)

    staging = staging_path(staging_dir, dataset_id, algorithm, fingerprint)
    baseline_path = staging / "baseline.joblib"
    candidate_path = staging / "candidate.joblib"
    manifest_path = staging / "train_manifest.json"
    if not (baseline_path.exists() and candidate_path.exists() and manifest_path.exists()):
        raise EvaluateGateError("EV_R01_GATE_STAGING_NOT_OK", f"incomplete staging at {staging}")

    baseline = _with_io_retry("load_baseline", lambda: joblib.load(baseline_path))
    candidate = _with_io_retry("load_candidate", lambda: joblib.load(candidate_path))

    test_path = datasets_dir / dataset_id / "test.parquet"
    test_df = _with_io_retry("read_test_parquet", lambda: pd.read_parquet(test_path)).reset_index(drop=True)

    y_true = test_df["label_score"].to_numpy()
    candidate_pred = np.array([clip_score(v) for v in candidate.predict(test_df)])
    baseline_pred = np.array([clip_score(v) for v in baseline.predict(test_df)])

    def _metrics(y_pred: np.ndarray) -> dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
        with warnings.catch_warnings():
            # The baseline's predictions are constant by definition
            # (DummyRegressor(mean)); scipy warns that spearman is undefined
            # for a constant input. Expected, already handled below (-> 0.0).
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            corr = spearmanr(y_true, y_pred).correlation
        spearman = float(corr) if corr is not None and not np.isnan(corr) else 0.0
        return {"mae": mae, "rmse": rmse, "spearman": spearman}

    candidate_metrics = _metrics(candidate_pred)
    baseline_metrics = _metrics(baseline_pred)

    mae_by_persona: dict[str, float] = {}
    for persona, group in test_df.groupby("persona"):
        idx = group.index.to_numpy()
        mae_by_persona[str(persona)] = float(mean_absolute_error(y_true[idx], candidate_pred[idx]))

    has_context = test_df["context_clean"].apply(len) > 0
    with_ctx_idx = test_df.index[has_context].to_numpy()
    no_ctx_idx = test_df.index[~has_context].to_numpy()

    def _safe_mae(idx: np.ndarray) -> float | None:
        if len(idx) == 0:
            return None
        return float(mean_absolute_error(y_true[idx], candidate_pred[idx]))

    context_breakdown = {
        "with_context_rows": len(with_ctx_idx),
        "with_context_mae": _safe_mae(with_ctx_idx),
        "no_context_rows": len(no_ctx_idx),
        "no_context_mae": _safe_mae(no_ctx_idx),
    }

    # Latência item a item — nunca em lote (EV-R06).
    latencies_ms: list[float] = []
    for i in range(len(test_df)):
        row = test_df.iloc[[i]]
        t0 = perf_counter()
        candidate.predict(row)
        latencies_ms.append((perf_counter() - t0) * 1000)
    latency = {
        "p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "sample_size": len(latencies_ms),
    }

    quality_gate_config = config.get("evaluate", {}).get("quality_gate", {})
    threshold_ratio = quality_gate_config.get("max_mae_ratio", 0.9)
    required_max_mae = threshold_ratio * baseline_metrics["mae"]
    gate_passed = candidate_metrics["mae"] <= required_max_mae

    eval_json = {
        "fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "test_rows": len(test_df),
        "metrics": {"candidate": candidate_metrics, "baseline": baseline_metrics},
        "mae_by_persona": mae_by_persona,
        "context_breakdown": context_breakdown,
        "latency": latency,
        "gate": {
            "threshold_ratio": threshold_ratio,
            "candidate_mae": candidate_metrics["mae"],
            "baseline_mae": baseline_metrics["mae"],
            "required_max_mae": required_max_mae,
            "passed": gate_passed,
        },
    }
    return EvalResult(eval_json=eval_json, gate_passed=gate_passed, staging_dir=staging, algorithm=algorithm)


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------


def _write_json_atomic(payload: dict[str, Any], dest: Path, run_id: str) -> None:
    _with_io_retry("write_eval_json", lambda: atomic_write(dest, lambda tmp: write_json(payload, tmp)), run_id)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evaluate.metrics")
    parser.add_argument("dataset_id")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--staging-dir", default="models/_staging")
    parser.add_argument("--datasets-dir", default="data/datasets")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    config = _load_config(Path(args.config))

    _log(logging.INFO, "evaluate_started", run_id=run_id, dataset_id=args.dataset_id)

    try:
        result = evaluate_candidate(
            args.dataset_id, config, staging_dir=args.staging_dir, datasets_dir=args.datasets_dir
        )
    except EvaluateGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except EvaluateIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    _log(logging.INFO, "staging_loaded", run_id=run_id, staging_dir=str(result.staging_dir))
    _log(
        logging.INFO,
        "metrics_computed",
        run_id=run_id,
        mae_candidate=result.eval_json["metrics"]["candidate"]["mae"],
        mae_baseline=result.eval_json["metrics"]["baseline"]["mae"],
        rmse_candidate=result.eval_json["metrics"]["candidate"]["rmse"],
        spearman_candidate=result.eval_json["metrics"]["candidate"]["spearman"],
        test_rows=result.eval_json["test_rows"],
    )
    personas = result.eval_json["mae_by_persona"]
    if personas:
        _log(
            logging.INFO,
            "mae_by_persona_computed",
            run_id=run_id,
            personas=list(personas),
            min_mae=min(personas.values()),
            max_mae=max(personas.values()),
        )
    _log(
        logging.INFO,
        "mae_by_context_computed",
        run_id=run_id,
        with_context_mae=result.eval_json["context_breakdown"]["with_context_mae"],
        no_context_mae=result.eval_json["context_breakdown"]["no_context_mae"],
        no_context_rows=result.eval_json["context_breakdown"]["no_context_rows"],
    )
    _log(logging.INFO, "latency_measured", run_id=run_id, **result.eval_json["latency"])

    finished_at = datetime.now(timezone.utc)
    eval_json = {
        **result.eval_json,
        "evaluated_at": now_iso(),
        "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
    }
    eval_json_path = result.staging_dir / "eval.json"

    try:
        _write_json_atomic(eval_json, eval_json_path, run_id)
    except EvaluateIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    if result.gate_passed:
        _log(
            logging.INFO,
            "quality_gate_passed",
            run_id=run_id,
            candidate_mae=eval_json["gate"]["candidate_mae"],
            required_max_mae=eval_json["gate"]["required_max_mae"],
        )
    else:
        _log(
            logging.ERROR,
            "quality_gate_failed",
            run_id=run_id,
            candidate_mae=eval_json["gate"]["candidate_mae"],
            baseline_mae=eval_json["gate"]["baseline_mae"],
            required_max_mae=eval_json["gate"]["required_max_mae"],
        )

    _log(
        logging.INFO,
        "evaluate_finished",
        run_id=run_id,
        duration_ms=eval_json["duration_ms"],
        eval_json_path=str(eval_json_path),
    )
    print(str(eval_json_path))
    return 0 if result.gate_passed else 5


if __name__ == "__main__":
    raise SystemExit(main())
