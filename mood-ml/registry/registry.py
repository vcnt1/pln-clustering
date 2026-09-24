"""[7] Registers immutable model versions and promotes them to active.json (see specs/07-registry.md)."""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from common.errors import PipelineError
from common.io import atomic_write, with_io_retry, write_json
from common.log import configure_logging, log, now_iso

logger = logging.getLogger("registry.registry")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RegisterGateError(PipelineError):
    pass


class CandidateRejectedError(PipelineError):
    pass


class PromoteGateError(PipelineError):
    pass


class MetricRegressionError(PipelineError):
    pass


class RegistryIOError(PipelineError):
    pass


_log = partial(log, logger)
_configure_logging = partial(configure_logging, logger)


def _with_io_retry(operation: str, func: Any, run_id: str, error_code: str) -> Any:
    return with_io_retry(operation, func, run_id, logger=logger, error=lambda detail: RegistryIOError(error_code, detail))


# ---------------------------------------------------------------------------
# register_model — resolve staging by glob -> F1 (gate) -> F2 (fingerprint
# scan) -> F3a/F3b (idempotent or mint) -> F4 (assemble manifest)
# ---------------------------------------------------------------------------


@dataclass
class RegisterResult:
    model_version: str
    manifest: dict[str, Any]
    candidate_path: Path | None
    reused: bool


def _resolve_staging(dataset_id: str, fingerprint: str, staging_dir: Path) -> Path:
    """Locates models/_staging/<dataset_id>/<algorithm>/<fingerprint>/ without
    knowing `algorithm` ahead of time: the fingerprint already encodes the
    algorithm choice (spec 06 §3.3), so it can only match one directory."""
    matches = sorted((staging_dir / dataset_id).glob(f"*/{fingerprint}")) if (staging_dir / dataset_id).exists() else []
    if len(matches) == 0:
        raise RegisterGateError(
            "RG_R01_GATE_STAGING_INCOMPLETE", f"no staging found for dataset_id={dataset_id} fingerprint={fingerprint}"
        )
    if len(matches) > 1:
        raise RuntimeError(f"fingerprint collision across algorithms: {matches}")
    return matches[0]


def _scan_existing_manifests(models_dir: Path) -> list[dict[str, Any]]:
    manifests = []
    for manifest_path in sorted(models_dir.glob("mood-*/manifest.json")):
        try:
            manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def _mint_model_version(models_dir: Path) -> str:
    now = datetime.now(timezone.utc)
    prefix = f"mood-{now.year:04d}.{now.month:02d}."
    max_n = -1
    if models_dir.exists():
        for candidate_dir in models_dir.glob(f"{prefix}*"):
            m = re.fullmatch(re.escape(prefix) + r"(\d+)", candidate_dir.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    return f"{prefix}{max_n + 1}"


def register_model(
    dataset_id: str,
    fingerprint: str,
    staging_dir: str | Path = Path("models/_staging"),
    models_dir: str | Path = Path("models"),
    datasets_dir: str | Path = Path("data/datasets"),
) -> RegisterResult:
    """Returns a RegisterResult. Writes nothing; does not decide an exit code
    (same separation as the other builders in this pipeline)."""
    staging_dir = Path(staging_dir)
    models_dir = Path(models_dir)
    datasets_dir = Path(datasets_dir)

    staging = _resolve_staging(dataset_id, fingerprint, staging_dir)

    eval_path = staging / "eval.json"
    train_manifest_path = staging / "train_manifest.json"
    candidate_path = staging / "candidate.joblib"
    if not (eval_path.exists() and train_manifest_path.exists() and candidate_path.exists()):
        raise RegisterGateError("RG_R01_GATE_STAGING_INCOMPLETE", f"incomplete staging at {staging}")

    eval_json = json.loads(eval_path.read_text(encoding="utf-8"))
    if not eval_json.get("gate", {}).get("passed"):
        raise CandidateRejectedError(
            "RG_R02_CANDIDATE_REJECTED", f"gate.passed is false for fingerprint {fingerprint}"
        )

    # F2 — varredura de idempotência
    for manifest in _scan_existing_manifests(models_dir):
        if manifest.get("training_fingerprint") == fingerprint:
            return RegisterResult(manifest["model_version"], manifest, None, reused=True)

    # F3b — minta model_version novo
    model_version = _mint_model_version(models_dir)

    # F4 — montagem do manifesto
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    dataset_json = json.loads((datasets_dir / dataset_id / "dataset.json").read_text(encoding="utf-8"))

    manifest = {
        "model_version": model_version,
        "scale": "-1 to 1",
        "mood_labels": None,
        "trained_at": train_manifest["trained_at"],
        "dataset_id": dataset_id,
        "label_set_id": dataset_json["label_set_id"],
        "feature_spec_version": dataset_json["feature_spec_version"],
        "history_window": 30,
        "history_scope": "conversation",
        "algorithm": train_manifest["algorithm"],
        "metrics": eval_json["metrics"]["candidate"],
        "code_commit": train_manifest["code_commit"],
        "training_fingerprint": fingerprint,
    }
    return RegisterResult(model_version, manifest, candidate_path, reused=False)


# ---------------------------------------------------------------------------
# promote_model — F1 (gate) -> F2 (já ativo?) -> F3 (comparação de métricas)
# ---------------------------------------------------------------------------


@dataclass
class PromoteResult:
    model_version: str
    already_active: bool
    regression: bool


def promote_model(
    model_version: str,
    models_dir: str | Path = Path("models"),
    force: bool = False,
) -> PromoteResult:
    """Returns a PromoteResult. Writes nothing; does not decide an exit code."""
    models_dir = Path(models_dir)
    manifest_path = models_dir / model_version / "manifest.json"
    if not manifest_path.exists():
        raise PromoteGateError("PM_R01_GATE_MODEL_NOT_FOUND", f"no manifest.json for {model_version}")
    target_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    active_path = models_dir / "active.json"
    if not active_path.exists():
        return PromoteResult(model_version, already_active=False, regression=False)

    active = json.loads(active_path.read_text(encoding="utf-8"))
    if active.get("model_version") == model_version:
        return PromoteResult(model_version, already_active=True, regression=False)

    active_manifest_path = models_dir / active["model_version"] / "manifest.json"
    if not active_manifest_path.exists():
        return PromoteResult(model_version, already_active=False, regression=False)

    active_manifest = json.loads(active_manifest_path.read_text(encoding="utf-8"))
    target_mae = target_manifest["metrics"]["mae"]
    active_mae = active_manifest["metrics"]["mae"]
    regression = target_mae > active_mae
    if regression and not force:
        raise MetricRegressionError(
            "PM_R03_METRIC_REGRESSION_BLOCKED",
            f"target MAE {target_mae} is worse than active MAE {active_mae}",
        )
    return PromoteResult(model_version, already_active=False, regression=regression)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _copy_atomic(src: Path, dest: Path, run_id: str) -> None:
    _with_io_retry("copy_model", lambda: atomic_write(dest, lambda tmp: shutil.copyfile(src, tmp)), run_id, "RG_R15_IO_FAILED")


def _write_json_atomic(payload: dict[str, Any], dest: Path, run_id: str, error_code: str) -> None:
    _with_io_retry("write_json", lambda: atomic_write(dest, lambda tmp: write_json(payload, tmp)), run_id, error_code)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m registry.registry")
    sub = parser.add_subparsers(dest="command", required=True)

    p_register = sub.add_parser("register")
    p_register.add_argument("dataset_id")
    p_register.add_argument("fingerprint")
    p_register.add_argument("--staging-dir", default="models/_staging")
    p_register.add_argument("--models-dir", default="models")
    p_register.add_argument("--datasets-dir", default="data/datasets")
    p_register.add_argument("--log-format", choices=["json", "text"], default="json")
    p_register.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("model_version")
    p_promote.add_argument("--force", action="store_true")
    p_promote.add_argument("--models-dir", default="models")
    p_promote.add_argument("--log-format", choices=["json", "text"], default="json")
    p_promote.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")

    return parser


def _run_register(args: argparse.Namespace) -> int:
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    _log(logging.INFO, "register_started", run_id=run_id, dataset_id=args.dataset_id, fingerprint=args.fingerprint)

    try:
        result = register_model(
            args.dataset_id,
            args.fingerprint,
            staging_dir=args.staging_dir,
            models_dir=args.models_dir,
            datasets_dir=args.datasets_dir,
        )
    except RegisterGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except CandidateRejectedError as exc:
        _log(
            logging.ERROR,
            "candidate_rejected",
            run_id=run_id,
            fingerprint=args.fingerprint,
            detail=exc.detail,
        )
        return 6
    except RegistryIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    if result.reused:
        _log(logging.INFO, "already_registered", run_id=run_id, model_version=result.model_version)
        print(result.model_version)
        return 0

    _log(logging.INFO, "model_version_minted", run_id=run_id, model_version=result.model_version)
    model_dir = Path(args.models_dir) / result.model_version

    try:
        _copy_atomic(result.candidate_path, model_dir / "model.joblib", run_id)
        _write_json_atomic(result.manifest, model_dir / "manifest.json", run_id, "RG_R15_IO_FAILED")
    except RegistryIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    _log(
        logging.INFO,
        "manifest_written",
        run_id=run_id,
        model_version=result.model_version,
        algorithm=result.manifest["algorithm"],
        metrics=result.manifest["metrics"],
    )
    _log(logging.INFO, "register_finished", run_id=run_id, model_version=result.model_version, duration_ms=duration_ms)
    print(result.model_version)
    return 0


def _run_promote(args: argparse.Namespace) -> int:
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    _log(logging.INFO, "promote_started", run_id=run_id, target_model_version=args.model_version)

    try:
        result = promote_model(args.model_version, models_dir=args.models_dir, force=args.force)
    except PromoteGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except MetricRegressionError as exc:
        _log(logging.WARNING, "promotion_blocked_regression", run_id=run_id, detail=exc.detail)
        return 7
    except RegistryIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    if result.already_active:
        _log(logging.INFO, "already_active", run_id=run_id, model_version=result.model_version)
        return 0

    active_json = {"model_version": result.model_version, "promoted_at": now_iso()}
    try:
        _write_json_atomic(active_json, Path(args.models_dir) / "active.json", run_id, "PM_R09_IO_FAILED")
    except RegistryIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    duration_ms = int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)
    _log(logging.INFO, "active_json_written", run_id=run_id, new_model_version=result.model_version)
    _log(logging.INFO, "promote_finished", run_id=run_id, model_version=result.model_version, duration_ms=duration_ms)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    if args.command == "register":
        return _run_register(args)
    return _run_promote(args)


if __name__ == "__main__":
    raise SystemExit(main())
