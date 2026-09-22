"""[9] Offline pipeline CLI: python -m pipeline <step> --config configs/pipeline.yaml (see specs/09-orquestracao-ci.md)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import evaluate.metrics as evaluate_metrics
import ingest.validate as ingest_validate
import labels.build as labels_build
import registry.registry as registry_registry
import train.train as train_train
import transform.split as transform_split
from transform.features import FEATURE_SPEC_VERSION

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Fingerprint recomputation e geração de dataset_id — a única lógica nova
# desta camada (D1/D2, specs/09-orquestracao-ci.md §2.2/§2.4)
# ---------------------------------------------------------------------------

_DATASET_ID_SUFFIXES = "abcdefghijklmnopqrstuvwxyz"


def _mint_dataset_id(datasets_dir: Path, today: date | None = None) -> str:
    """PL-R05: ds-AAAA-MM-DD-<letra>, incrementando a letra (a..z) se o
    diretório candidato já existir. `today` existe só para testabilidade —
    em produção é sempre a data corrente em UTC."""
    day = today if today is not None else datetime.now(timezone.utc).date()
    prefix = f"ds-{day.year:04d}-{day.month:02d}-{day.day:02d}-"
    for letter in _DATASET_ID_SUFFIXES:
        candidate = f"{prefix}{letter}"
        if not (datasets_dir / candidate).exists():
            return candidate
    raise RuntimeError(f"exhausted dataset_id suffixes for {prefix}*")


def _recompute_fingerprint(dataset_id: str, config: dict[str, Any], datasets_dir: Path) -> str:
    """PL-R09: mesma fórmula que train.train usa internamente (spec 06
    §3.3), via as funções já públicas de evaluate.metrics — nunca lê o
    valor de volta de stdout/log de um passo anterior."""
    train_cfg = dict(config.get("train", {}))
    algorithm = train_cfg.pop("algorithm", "tfidf-ridge")
    dataset_sha256 = evaluate_metrics.dataset_parquet_hashes(datasets_dir, dataset_id)
    return evaluate_metrics.compute_training_fingerprint(dataset_sha256, algorithm, train_cfg, FEATURE_SPEC_VERSION)


def _find_model_version_by_fingerprint(models_dir: Path, fingerprint: str) -> str | None:
    """Só para o evento de log pipeline_finished (§5.2) ficar completo — não
    afeta o resultado de `all` (register já decidiu tudo antes disso).
    Varredura local, própria de pipeline.py, não o scan privado de
    registry.py."""
    for manifest_path in sorted(Path(models_dir).glob("mood-*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("training_fingerprint") == fingerprint:
            return manifest.get("model_version")
    return None


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Subcomandos individuais — wrappers finos sobre o main(argv) já existente
# de cada módulo (PL-R02, PL-R03): nenhuma lógica de porta/erro reimplementada.
# ---------------------------------------------------------------------------


def _run_ingest(args: argparse.Namespace) -> int:
    argv = [args.path, "--report-dir", args.report_dir, "--log-format", args.log_format, "--log-level", args.log_level]
    if args.skip_composition:
        argv.append("--skip-composition")
    if args.review_sample is not None:
        argv.extend(["--review-sample", str(args.review_sample)])
    return ingest_validate.main(argv)


def _run_labels(args: argparse.Namespace) -> int:
    argv = [
        args.path,
        "--report-dir", args.report_dir,
        "--labels-dir", args.labels_dir,
        "--log-format", args.log_format,
        "--log-level", args.log_level,
    ]
    return labels_build.main(argv)


def _run_split(args: argparse.Namespace) -> int:
    argv = [
        args.path,
        "--dataset-id", args.dataset_id,
        "--config", args.config,
        "--report-dir", args.report_dir,
        "--datasets-dir", args.datasets_dir,
        "--log-format", args.log_format,
        "--log-level", args.log_level,
    ]
    return transform_split.main(argv)


def _run_train(args: argparse.Namespace) -> int:
    argv = [
        args.dataset_id,
        "--config", args.config,
        "--datasets-dir", args.datasets_dir,
        "--staging-dir", args.staging_dir,
        "--log-format", args.log_format,
        "--log-level", args.log_level,
    ]
    return train_train.main(argv)


def _run_evaluate(args: argparse.Namespace) -> int:
    argv = [
        args.dataset_id,
        "--config", args.config,
        "--staging-dir", args.staging_dir,
        "--datasets-dir", args.datasets_dir,
        "--log-format", args.log_format,
        "--log-level", args.log_level,
    ]
    return evaluate_metrics.main(argv)


def _run_register(args: argparse.Namespace) -> int:
    argv = [
        "register",
        args.dataset_id,
        args.fingerprint,
        "--staging-dir", args.staging_dir,
        "--models-dir", args.models_dir,
        "--datasets-dir", args.datasets_dir,
        "--log-format", args.log_format,
        "--log-level", args.log_level,
    ]
    return registry_registry.main(argv)


def _run_promote(args: argparse.Namespace) -> int:
    argv = ["promote", args.model_version, "--models-dir", args.models_dir, "--log-format", args.log_format, "--log-level", args.log_level]
    if args.force:
        argv.append("--force")
    return registry_registry.main(argv)


# ---------------------------------------------------------------------------
# all — encadeia ingest -> labels -> split -> train -> evaluate -> register,
# falha rápida propagando o exit code do passo que falhou (PL-R06 a PL-R09)
# ---------------------------------------------------------------------------


def _run_all(args: argparse.Namespace) -> int:
    run_id = uuid.uuid4().hex[:8]
    t0 = time.perf_counter()
    _log(logging.INFO, "pipeline_started", run_id=run_id, subcommand="all")

    config = _load_config(Path(args.config))
    datasets_dir = Path(args.datasets_dir)
    models_dir = Path(args.models_dir)

    def _step(name: str, code: int) -> int | None:
        if code != 0:
            _log(logging.ERROR, "step_failed", run_id=run_id, step=name, exit_code=code)
            return code
        _log(logging.INFO, "step_finished", run_id=run_id, step=name, exit_code=0)
        return None

    _log(logging.INFO, "step_started", run_id=run_id, step="ingest")
    ingest_argv = [args.path, "--report-dir", args.report_dir]
    if args.skip_composition:
        ingest_argv.append("--skip-composition")
    failed = _step("ingest", ingest_validate.main(ingest_argv))
    if failed is not None:
        return failed

    _log(logging.INFO, "step_started", run_id=run_id, step="labels")
    labels_argv = [args.path, "--report-dir", args.report_dir, "--labels-dir", args.labels_dir]
    failed = _step("labels", labels_build.main(labels_argv))
    if failed is not None:
        return failed

    dataset_id = _mint_dataset_id(datasets_dir)
    _log(logging.INFO, "dataset_id_generated", run_id=run_id, dataset_id=dataset_id)

    _log(logging.INFO, "step_started", run_id=run_id, step="split")
    split_argv = [
        args.path,
        "--dataset-id", dataset_id,
        "--config", args.config,
        "--report-dir", args.report_dir,
        "--datasets-dir", args.datasets_dir,
    ]
    failed = _step("split", transform_split.main(split_argv))
    if failed is not None:
        return failed

    _log(logging.INFO, "step_started", run_id=run_id, step="train")
    train_argv = [dataset_id, "--config", args.config, "--datasets-dir", args.datasets_dir, "--staging-dir", args.staging_dir]
    failed = _step("train", train_train.main(train_argv))
    if failed is not None:
        return failed

    fingerprint = _recompute_fingerprint(dataset_id, config, datasets_dir)

    _log(logging.INFO, "step_started", run_id=run_id, step="evaluate")
    evaluate_argv = [
        dataset_id,
        "--config", args.config,
        "--staging-dir", args.staging_dir,
        "--datasets-dir", args.datasets_dir,
    ]
    failed = _step("evaluate", evaluate_metrics.main(evaluate_argv))
    if failed is not None:
        # Inclui o quality gate reprovado (exit 5, spec 06): register nunca
        # roda sobre um candidato que não passou no gate (PL-R08).
        return failed

    _log(logging.INFO, "step_started", run_id=run_id, step="register")
    register_argv = [
        "register",
        dataset_id,
        fingerprint,
        "--staging-dir", args.staging_dir,
        "--models-dir", args.models_dir,
        "--datasets-dir", args.datasets_dir,
    ]
    failed = _step("register", registry_registry.main(register_argv))
    if failed is not None:
        return failed

    model_version = _find_model_version_by_fingerprint(models_dir, fingerprint)
    duration_ms = int((time.perf_counter() - t0) * 1000)
    _log(
        logging.INFO,
        "pipeline_finished",
        run_id=run_id,
        corpus_id=Path(args.path).stem,
        dataset_id=dataset_id,
        fingerprint=fingerprint,
        model_version=model_version,
        duration_ms=duration_ms,
    )
    return 0


# ---------------------------------------------------------------------------
# Logging — mesmo padrão duplicado por módulo das Fases 1-4, eventos só de
# orquestração (§5.2): cada passo já loga por conta própria ao chamar o
# main() daquele módulo.
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": _now_iso(), "level": record.levelname, "event": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "fields", None)
        suffix = f" {extra}" if extra else ""
        return f"{record.levelname:<7} {record.getMessage()}{suffix}"


def _configure_logging(log_format: str, log_level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if log_format == "json" else _TextFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False


def _log(level: int, event: str, **fields: Any) -> None:
    run_id = fields.pop("run_id", None)
    logger.log(level, event, extra={"fields": {"run_id": run_id, **fields}})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_log_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("path")
    p_ingest.add_argument("--report-dir", default="data/reports")
    p_ingest.add_argument("--skip-composition", action="store_true")
    p_ingest.add_argument("--review-sample", type=int, default=None, metavar="N")
    _add_common_log_args(p_ingest)

    p_labels = sub.add_parser("labels")
    p_labels.add_argument("path")
    p_labels.add_argument("--report-dir", default="data/reports")
    p_labels.add_argument("--labels-dir", default="data/labels")
    _add_common_log_args(p_labels)

    p_split = sub.add_parser("split")
    p_split.add_argument("path")
    p_split.add_argument("--dataset-id", required=True)
    p_split.add_argument("--config", default="configs/pipeline.yaml")
    p_split.add_argument("--report-dir", default="data/reports")
    p_split.add_argument("--datasets-dir", default="data/datasets")
    _add_common_log_args(p_split)

    p_train = sub.add_parser("train")
    p_train.add_argument("dataset_id")
    p_train.add_argument("--config", default="configs/pipeline.yaml")
    p_train.add_argument("--datasets-dir", default="data/datasets")
    p_train.add_argument("--staging-dir", default="models/_staging")
    _add_common_log_args(p_train)

    p_evaluate = sub.add_parser("evaluate")
    p_evaluate.add_argument("dataset_id")
    p_evaluate.add_argument("--config", default="configs/pipeline.yaml")
    p_evaluate.add_argument("--staging-dir", default="models/_staging")
    p_evaluate.add_argument("--datasets-dir", default="data/datasets")
    _add_common_log_args(p_evaluate)

    p_register = sub.add_parser("register")
    p_register.add_argument("dataset_id")
    p_register.add_argument("fingerprint")
    p_register.add_argument("--staging-dir", default="models/_staging")
    p_register.add_argument("--models-dir", default="models")
    p_register.add_argument("--datasets-dir", default="data/datasets")
    _add_common_log_args(p_register)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("model_version")
    p_promote.add_argument("--force", action="store_true")
    p_promote.add_argument("--models-dir", default="models")
    _add_common_log_args(p_promote)

    p_all = sub.add_parser("all")
    p_all.add_argument("path")
    p_all.add_argument("--skip-composition", action="store_true")
    p_all.add_argument("--config", default="configs/pipeline.yaml")
    p_all.add_argument("--report-dir", default="data/reports")
    p_all.add_argument("--labels-dir", default="data/labels")
    p_all.add_argument("--datasets-dir", default="data/datasets")
    p_all.add_argument("--staging-dir", default="models/_staging")
    p_all.add_argument("--models-dir", default="models")
    _add_common_log_args(p_all)

    return parser


_HANDLERS = {
    "ingest": _run_ingest,
    "labels": _run_labels,
    "split": _run_split,
    "train": _run_train,
    "evaluate": _run_evaluate,
    "register": _run_register,
    "promote": _run_promote,
    "all": _run_all,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    return _HANDLERS[args.subcommand](args)


if __name__ == "__main__":
    raise SystemExit(main())
