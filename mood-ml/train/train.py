"""[5] Trains the baseline and candidate models on a dataset's train split (see specs/06-train-evaluate.md)."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from common.errors import PipelineError
from common.io import atomic_write, atomic_write_dir, with_io_retry, write_json_atomic
from common.log import configure_logging, log, now_iso
from evaluate.metrics import (
    EMBEDDINGS_FT,
    SUPPORTED_ALGORITHMS,
    clip_score,
    dataset_parquet_hashes,
    staging_path,
    training_identity,
)
from transform.features import FEATURE_SPEC_VERSION, join_context_list

logger = logging.getLogger("train.train")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TrainGateError(PipelineError):
    pass


class TrainSanityError(PipelineError):
    pass


class TrainConfigError(PipelineError):
    pass


class TrainIOError(PipelineError):
    pass


_log = partial(log, logger)
_configure_logging = partial(configure_logging, logger)
_with_io_retry = partial(with_io_retry, logger=logger, error=lambda detail: TrainIOError("TN_IO_FAILED", detail))


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, never blocks training
        return "unknown"


# ---------------------------------------------------------------------------
# Vetorização (ADR-0008, abordagem A) — spec 06 §3.5
# ---------------------------------------------------------------------------


def _build_pipeline(hyperparameters: dict[str, Any]) -> Pipeline:
    ridge_cfg = hyperparameters.get("ridge", {})
    word_cfg = hyperparameters.get("tfidf_word", {})
    char_cfg = hyperparameters.get("tfidf_char", {})
    context_cfg = hyperparameters.get("context", {})

    text_union = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=tuple(word_cfg.get("ngram_range", [1, 2])))),
            (
                "char",
                TfidfVectorizer(
                    ngram_range=tuple(char_cfg.get("ngram_range", [2, 5])),
                    analyzer=char_cfg.get("analyzer", "char_wb"),
                ),
            ),
        ]
    )

    context_pipeline = Pipeline(
        [
            ("join", FunctionTransformer(join_context_list)),
            ("tfidf", TfidfVectorizer(ngram_range=tuple(context_cfg.get("ngram_range", [1, 2])))),
        ]
    )

    features = ColumnTransformer(
        transformers=[
            ("text", text_union, "text_clean"),
            ("context", context_pipeline, "context_clean"),
        ],
        transformer_weights={"context": context_cfg.get("weight", 0.5)},
    )

    ridge = Ridge(alpha=ridge_cfg.get("alpha", 1.0), solver=ridge_cfg.get("solver", "lsqr"))

    return Pipeline([("features", features), ("ridge", ridge)])


# ---------------------------------------------------------------------------
# F1b — abordagem C (spec 11): valida config e dependências antes de ler
# qualquer .parquet. TN_R17 já filtrou algoritmos desconhecidos; aqui só
# embeddings-ft chega.
# ---------------------------------------------------------------------------


_VALID_DEVICES = {"auto", "cpu", "cuda"}
_VALID_RIDGE_SOLVERS = {"auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"}


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_positive_int_or_none(value: Any) -> bool:
    return value is None or _is_positive_int(value)


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _is_nonneg_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def _is_unit_interval(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


# FT-R01: cada chave listada abaixo precisa existir E ter um valor que passe no
# predicado — "malformada" (spec 11 §7) cobre os dois casos. As chaves obrigatórias
# vêm de `validators.keys()`, uma única fonte de verdade (nunca uma lista de nomes
# separada da checagem de valor).
_EMBEDDINGS_FIELD_VALIDATORS: dict[str, dict[str, Callable[[Any], bool]]] = {
    "encoder": {
        "source": _is_nonempty_str,
        "revision": _is_nonempty_str,
        "max_length": _is_positive_int,
        "device": lambda value: value in _VALID_DEVICES,
        "num_threads": _is_positive_int_or_none,
    },
    "context": {
        "weight": _is_nonneg_number,
    },
    "probe": {
        "alpha": _is_positive_number,
        "solver": lambda value: value in _VALID_RIDGE_SOLVERS,
    },
    "finetune": {
        "epochs_max": _is_positive_int,
        "patience": _is_nonneg_int,
        "batch_size": _is_positive_int,
        "lr_encoder": _is_positive_number,
        "lr_head": _is_positive_number,
        "weight_decay": _is_nonneg_number,
        "warmup_ratio": _is_unit_interval,
        "dropout": _is_unit_interval,
        "freeze_word_embeddings": _is_bool,
    },
}


def _check_embeddings_revision(config: dict[str, Any]) -> None:
    """FT-R04: `revision` ausente ou vazia aborta antes de qualquer outra
    checagem de C — é o jeito mais barato de falhar rápido quando alguém
    esquece de fixar o SHA do encoder no hub."""
    embeddings_config = config.get("train", {}).get("embeddings", {})
    encoder_config = embeddings_config.get("encoder", {}) if isinstance(embeddings_config, dict) else {}
    revision = encoder_config.get("revision") if isinstance(encoder_config, dict) else None
    if not revision:
        raise TrainConfigError(
            "FT_R04_REVISION_MISSING", "train.embeddings.encoder.revision is missing or empty"
        )


def _check_embeddings_deps() -> None:
    """FT-R02: torch/transformers são dependências só de C (requirements-embeddings.txt,
    Fase 3); a suíte de A nunca os importa."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise TrainConfigError("FT_R02_DEPS_MISSING", f"torch/transformers not installed: {exc}") from exc


def _check_embeddings_config(config: dict[str, Any]) -> None:
    """FT-R01: toda chave de `train.embeddings.<bloco>` listada em
    `_EMBEDDINGS_FIELD_VALIDATORS` precisa existir e ter um valor válido —
    sem default no código."""
    embeddings_config = config.get("train", {}).get("embeddings")
    if not isinstance(embeddings_config, dict):
        raise TrainConfigError("FT_R01_CONFIG_INVALID", "train.embeddings is missing or not a mapping")

    for block, validators in _EMBEDDINGS_FIELD_VALIDATORS.items():
        block_config = embeddings_config.get(block)
        if not isinstance(block_config, dict):
            raise TrainConfigError(
                "FT_R01_CONFIG_INVALID", f"train.embeddings.{block} is missing or not a mapping"
            )
        missing = [key for key in validators if key not in block_config]
        if missing:
            raise TrainConfigError(
                "FT_R01_CONFIG_INVALID", f"train.embeddings.{block} missing required keys: {missing}"
            )
        invalid = [key for key, is_valid in validators.items() if not is_valid(block_config[key])]
        if invalid:
            raise TrainConfigError(
                "FT_R01_CONFIG_INVALID", f"train.embeddings.{block} has invalid values for: {invalid}"
            )


def _fit_embeddings_ft(train_df: pd.DataFrame, validation_df: pd.DataFrame, hyperparameters: dict[str, Any]) -> Any:
    """Ramo C (spec 11 §3): baixa o encoder base, roda sondagem + fine-tuning,
    devolve `(candidate, info)` — `candidate` é um `EmbeddingMoodModel` já
    vinculado ao encoder treinado (pronto para `.predict()`); `info` alimenta
    `train_manifest.json` (FT-R14). Imports de torch/transform.embeddings/
    train.embeddings_ft são *lazy*: só quem pede `algorithm=embeddings-ft`
    paga esse custo (FT-R02) — os módulos de A nunca importam torch."""
    from train.embeddings_ft import FineTuneSettings, NonFiniteError, fit_embeddings_ft
    from transform.embeddings import (
        EmbeddingMoodModel,
        EncoderSettings,
        load_base_encoder,
        resolve_device,
    )

    embeddings_config = hyperparameters["embeddings"]
    encoder_config = embeddings_config["encoder"]
    encoder_settings = EncoderSettings(
        source=encoder_config["source"],
        revision=encoder_config["revision"],
        max_length=encoder_config["max_length"],
        device=encoder_config["device"],
        num_threads=encoder_config["num_threads"],
    )
    context_weight = embeddings_config["context"]["weight"]
    probe_config = embeddings_config["probe"]
    finetune_config = embeddings_config["finetune"]
    finetune_settings = FineTuneSettings(
        epochs_max=finetune_config["epochs_max"],
        patience=finetune_config["patience"],
        batch_size=finetune_config["batch_size"],
        lr_encoder=finetune_config["lr_encoder"],
        lr_head=finetune_config["lr_head"],
        weight_decay=finetune_config["weight_decay"],
        warmup_ratio=finetune_config["warmup_ratio"],
        dropout=finetune_config["dropout"],
        freeze_word_embeddings=finetune_config["freeze_word_embeddings"],
    )
    device = resolve_device(encoder_settings.device)

    try:
        encoder, tokenizer = load_base_encoder(encoder_settings)
    except Exception as exc:
        raise TrainGateError("FT_ENCODER_UNAVAILABLE", f"failed to download base encoder: {exc}") from exc

    try:
        coef, intercept, info = fit_embeddings_ft(
            encoder,
            tokenizer,
            train_df,
            validation_df,
            encoder_settings.max_length,
            context_weight,
            probe_config["alpha"],
            probe_config["solver"],
            finetune_settings,
            hyperparameters["seed"],
            device,
        )
    except NonFiniteError as exc:
        raise TrainSanityError(exc.code, exc.detail) from exc

    candidate = EmbeddingMoodModel(encoder_settings, context_weight, coef, intercept).bind(encoder, tokenizer)
    return candidate, info


def _save_encoder_atomic(candidate: Any, dest: Path, run_id: str) -> None:
    _with_io_retry("write_encoder", lambda: atomic_write_dir(dest, candidate.save_encoder), run_id)


# ---------------------------------------------------------------------------
# build_candidate — F1 (gate) -> F1b (config de C) -> F2 (fingerprint/idempotency)
# -> F3 (load) -> F4 (vectorize + fit) -> F5 (sanity)
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    baseline: Any
    candidate: Any
    train_manifest: dict[str, Any]
    staging_dir: Path
    reused: bool


def build_candidate(
    dataset_id: str,
    config: dict[str, Any],
    datasets_dir: str | Path = Path("data/datasets"),
    staging_dir: str | Path = Path("models/_staging"),
    run_id: str | None = None,
) -> TrainResult:
    """Returns a TrainResult. Writes nothing; does not decide an exit code
    (same separation as validate_corpus/build_labels/build_dataset)."""
    datasets_dir = Path(datasets_dir)
    staging_dir = Path(staging_dir)

    # F1 — portão
    dataset_json_path = datasets_dir / dataset_id / "dataset.json"
    if not dataset_json_path.exists():
        raise TrainGateError("TN_R01_GATE_DATASET_NOT_OK", f"dataset.json not found for {dataset_id}")
    try:
        dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
        test_rows = dataset_json["row_counts"]["test"]["rows"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TrainGateError(
            "TN_R01_GATE_DATASET_NOT_OK", f"dataset.json unreadable or incomplete for {dataset_id}: {exc!r}"
        ) from exc
    for split in ("train", "validation", "test"):
        if not (datasets_dir / dataset_id / f"{split}.parquet").exists():
            raise TrainGateError("TN_R01_GATE_DATASET_NOT_OK", f"{split}.parquet missing for {dataset_id}")

    algorithm = config.get("train", {}).get("algorithm", "tfidf-ridge")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise TrainConfigError(
            "TN_R17_UNSUPPORTED_ALGORITHM",
            f"unsupported algorithm: {algorithm!r} (only {sorted(SUPPORTED_ALGORITHMS)} implemented)",
        )

    if algorithm == EMBEDDINGS_FT:
        _check_embeddings_revision(config)
        _check_embeddings_deps()
        _check_embeddings_config(config)

    # F2 — identidade e idempotência
    dataset_sha256 = dataset_parquet_hashes(datasets_dir, dataset_id)
    algorithm, hyperparameters, fingerprint = training_identity(config, dataset_sha256)
    staging = staging_path(staging_dir, dataset_id, algorithm, fingerprint)

    baseline_path = staging / "baseline.joblib"
    candidate_path = staging / "candidate.joblib"
    manifest_path = staging / "train_manifest.json"
    encoder_path = staging / "encoder"

    staging_complete = baseline_path.exists() and candidate_path.exists() and manifest_path.exists()
    if algorithm == EMBEDDINGS_FT:
        staging_complete = staging_complete and encoder_path.exists()

    if staging_complete:
        baseline = _with_io_retry("load_baseline", lambda: joblib.load(baseline_path), run_id)
        candidate = _with_io_retry("load_candidate", lambda: joblib.load(candidate_path), run_id)
        if algorithm == EMBEDDINGS_FT:
            candidate.attach(encoder_path)
        train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return TrainResult(baseline, candidate, train_manifest, staging, reused=True)

    # F3 — carga
    train_df = _with_io_retry(
        "read_train_parquet", lambda: pd.read_parquet(datasets_dir / dataset_id / "train.parquet"), run_id
    )
    validation_df = _with_io_retry(
        "read_validation_parquet", lambda: pd.read_parquet(datasets_dir / dataset_id / "validation.parquet"), run_id
    )
    y_train = train_df["label_score"].to_numpy()

    # F4 — vetorização + fit
    extra_manifest_fields: dict[str, Any] = {}
    if algorithm == EMBEDDINGS_FT:
        candidate, embeddings_info = _fit_embeddings_ft(train_df, validation_df, hyperparameters)
        extra_manifest_fields = embeddings_info
    else:
        candidate = _build_pipeline(hyperparameters)
        candidate.fit(train_df, y_train)

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(train_df, y_train)

    # F5 — sanity: previsões finitas sobre o próprio train
    train_preds = candidate.predict(train_df)
    if not np.all(np.isfinite(train_preds)):
        code = "FT_R13_NON_FINITE" if algorithm == EMBEDDINGS_FT else "TN_R14_NON_FINITE_PREDICTION"
        raise TrainSanityError(code, "candidate produced a non-finite prediction on its own train split")

    # Diagnóstico de validation (informativo, não decide nada — §3.4)
    validation_preds = np.array([clip_score(v) for v in candidate.predict(validation_df)])
    validation_mae = float(np.mean(np.abs(validation_preds - validation_df["label_score"].to_numpy())))

    train_manifest = {
        "fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "algorithm": algorithm,
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "hyperparameters": hyperparameters,
        "row_counts": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": test_rows,
        },
        "validation_mae_candidate": validation_mae,
        "code_commit": _git_commit(),
        **extra_manifest_fields,
    }
    return TrainResult(baseline, candidate, train_manifest, staging, reused=False)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _dump_joblib_atomic(obj: Any, dest: Path, run_id: str) -> None:
    _with_io_retry(f"write_{dest.stem}", lambda: atomic_write(dest, lambda tmp: joblib.dump(obj, tmp)), run_id)


def _write_json_atomic(payload: dict[str, Any], dest: Path, run_id: str) -> None:
    _with_io_retry("write_train_manifest", lambda: write_json_atomic(payload, dest), run_id)


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m train.train")
    parser.add_argument("dataset_id")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--datasets-dir", default="data/datasets")
    parser.add_argument("--staging-dir", default="models/_staging")
    parser.add_argument("--log-format", choices=["json", "text"], default="json")
    parser.add_argument("--log-level", choices=["INFO", "WARNING", "ERROR"], default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_format, args.log_level)
    run_id = uuid.uuid4().hex[:8]
    started_at = datetime.now(timezone.utc)
    config = _load_config(Path(args.config))
    algorithm = config.get("train", {}).get("algorithm", "tfidf-ridge")

    _log(logging.INFO, "train_started", run_id=run_id, dataset_id=args.dataset_id, algorithm=algorithm)

    try:
        result = build_candidate(
            args.dataset_id, config, datasets_dir=args.datasets_dir, staging_dir=args.staging_dir, run_id=run_id
        )
    except TrainGateError as exc:
        _log(logging.ERROR, "gate_blocked", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 3
    except TrainSanityError as exc:
        _log(logging.ERROR, "sanity_check_failed", run_id=run_id, model="candidate", reason=exc.code, detail=exc.detail)
        return 4
    except TrainConfigError as exc:
        _log(logging.ERROR, "config_invalid", run_id=run_id, reason=exc.code, detail=exc.detail)
        return 2
    except TrainIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all for INTERNAL_ERROR (exit 1)
        _log(logging.ERROR, "internal_error", run_id=run_id, exc_type=type(exc).__name__, detail=str(exc))
        return 1

    if result.reused:
        _log(logging.INFO, "already_trained", run_id=run_id, fingerprint=result.train_manifest["fingerprint"])
        print(str(result.staging_dir))
        return 0

    _log(logging.INFO, "fingerprint_computed", run_id=run_id, fingerprint=result.train_manifest["fingerprint"], reused=False)
    _log(
        logging.INFO,
        "data_loaded",
        run_id=run_id,
        train_rows=result.train_manifest["row_counts"]["train"],
        validation_rows=result.train_manifest["row_counts"]["validation"],
    )
    _log(logging.INFO, "fit_finished", run_id=run_id, model="baseline")
    _log(logging.INFO, "fit_finished", run_id=run_id, model="candidate")
    _log(
        logging.INFO,
        "validation_diagnostic",
        run_id=run_id,
        mae_candidate_on_validation=result.train_manifest["validation_mae_candidate"],
    )
    _log(logging.INFO, "sanity_check_passed", run_id=run_id, checks=["finite_predictions"])

    finished_at = datetime.now(timezone.utc)
    train_manifest = {
        **result.train_manifest,
        "trained_at": now_iso(),
        "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
    }

    try:
        _dump_joblib_atomic(result.baseline, result.staging_dir / "baseline.joblib", run_id)
        if algorithm == EMBEDDINGS_FT:
            _save_encoder_atomic(result.candidate, result.staging_dir / "encoder", run_id)
        _dump_joblib_atomic(result.candidate, result.staging_dir / "candidate.joblib", run_id)
        _write_json_atomic(train_manifest, result.staging_dir / "train_manifest.json", run_id)
    except TrainIOError as exc:
        _log(logging.ERROR, "io_failed", run_id=run_id, detail=exc.detail)
        return 1

    _log(
        logging.INFO,
        "train_finished",
        run_id=run_id,
        fingerprint=train_manifest["fingerprint"],
        staging_path=str(result.staging_dir),
        duration_ms=train_manifest["duration_ms"],
    )
    print(str(result.staging_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
