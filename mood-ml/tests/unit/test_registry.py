import json
from pathlib import Path

import pytest

from registry.registry import (
    CandidateRejectedError,
    MetricRegressionError,
    RegisterGateError,
    _mint_model_version,
    _write_json_atomic,
    promote_model,
    register_model,
)
from tests.conftest import build_synthetic_dataset, evaluate_staging, train_and_stage


def _build_ready_staging(directory: Path, dataset_id: str, config: dict, signal: str = "strong") -> str:
    """Runs the real train -> evaluate chain and returns the resulting
    fingerprint, with a fully-written staging dir (candidate.joblib,
    train_manifest.json, eval.json) ready for register_model."""
    build_synthetic_dataset(directory, dataset_id, signal=signal)
    train_and_stage(directory, dataset_id, config)
    eval_json = evaluate_staging(directory, dataset_id, config)
    return eval_json["fingerprint"]


def _register(directory: Path, dataset_id: str, fingerprint: str):
    return register_model(
        dataset_id,
        fingerprint,
        staging_dir=directory / "staging",
        models_dir=directory / "models",
        datasets_dir=directory / "datasets",
    )


def _write_registered_manifest(directory: Path, result) -> Path:
    """Mirrors the CLI's write step (registry.registry._run_register), so
    idempotency/sequencing tests can build up models_dir state without going
    through the CLI's argv plumbing."""
    model_dir = directory / "models" / result.model_version
    _write_json_atomic(result.manifest, model_dir / "manifest.json", "test", "RG_R15_IO_FAILED")
    return model_dir / "manifest.json"


# ---------------------------------------------------------------------------
# Resolução do staging por glob (lacuna #1) — algorithm nunca é passado
# explicitamente, é recuperado a partir do fingerprint.
# ---------------------------------------------------------------------------


def test_resolve_staging_by_glob_without_knowing_algorithm(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _build_ready_staging(corpus_root, "ds-a", train_config, signal="strong")
    result = _register(corpus_root, "ds-a", fingerprint)
    assert result.manifest["algorithm"] == "tfidf-ridge"
    assert result.manifest["training_fingerprint"] == fingerprint


def test_gate_blocked_when_no_staging_matches_fingerprint(corpus_root: Path) -> None:
    with pytest.raises(RegisterGateError) as exc:
        _register(corpus_root, "ds-missing", "0" * 16)
    assert exc.value.code == "RG_R01_GATE_STAGING_INCOMPLETE"


def test_candidate_rejected_when_quality_gate_failed(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _build_ready_staging(corpus_root, "ds-b", train_config, signal="none")
    with pytest.raises(CandidateRejectedError) as exc:
        _register(corpus_root, "ds-b", fingerprint)
    assert exc.value.code == "RG_R02_CANDIDATE_REJECTED"


# ---------------------------------------------------------------------------
# Idempotência por training_fingerprint
# ---------------------------------------------------------------------------


def test_registration_is_idempotent_by_training_fingerprint(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _build_ready_staging(corpus_root, "ds-c", train_config, signal="strong")

    result1 = _register(corpus_root, "ds-c", fingerprint)
    assert result1.reused is False
    _write_registered_manifest(corpus_root, result1)

    result2 = _register(corpus_root, "ds-c", fingerprint)
    assert result2.reused is True
    assert result2.model_version == result1.model_version
    assert result2.candidate_path is None


# ---------------------------------------------------------------------------
# Numeração sequencial de model_version
# ---------------------------------------------------------------------------


def test_mint_model_version_starts_at_zero(tmp_path: Path) -> None:
    version = _mint_model_version(tmp_path / "models")
    assert version.endswith(".0")


def test_mint_model_version_increments_sequentially(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    first = _mint_model_version(models_dir)
    (models_dir / first).mkdir(parents=True)
    (models_dir / first / "manifest.json").write_text("{}", encoding="utf-8")

    second = _mint_model_version(models_dir)
    assert second != first
    first_n = int(first.rsplit(".", 1)[1])
    second_n = int(second.rsplit(".", 1)[1])
    assert second_n == first_n + 1


def test_two_different_fingerprints_get_sequential_versions(corpus_root: Path, train_config: dict) -> None:
    fp_a = _build_ready_staging(corpus_root, "ds-d", train_config, signal="strong")
    result_a = _register(corpus_root, "ds-d", fp_a)
    _write_registered_manifest(corpus_root, result_a)

    other_config = {**train_config, "train": {**train_config["train"], "ridge": {"alpha": 5.0, "solver": "lsqr"}}}
    fp_b = _build_ready_staging(corpus_root, "ds-e", other_config, signal="strong")
    result_b = _register(corpus_root, "ds-e", fp_b)

    assert result_a.model_version != result_b.model_version


# ---------------------------------------------------------------------------
# Campos fixos no manifesto (CA-06)
# ---------------------------------------------------------------------------


def test_manifest_has_fixed_history_and_null_mood_labels(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _build_ready_staging(corpus_root, "ds-f", train_config, signal="strong")
    result = _register(corpus_root, "ds-f", fingerprint)

    assert result.manifest["history_window"] == 30
    assert result.manifest["history_scope"] == "conversation"
    assert result.manifest["mood_labels"] is None
    assert result.manifest["scale"] == "-1 to 1"


# ---------------------------------------------------------------------------
# promote_model — comparação de métrica
# ---------------------------------------------------------------------------


def _write_manifest_with_mae(directory: Path, model_version: str, mae: float) -> None:
    manifest = {
        "model_version": model_version,
        "scale": "-1 to 1",
        "mood_labels": None,
        "history_window": 30,
        "history_scope": "conversation",
        "algorithm": "tfidf-ridge",
        "metrics": {"mae": mae, "rmse": mae, "spearman": 0.5},
        "training_fingerprint": f"fp-{model_version}",
    }
    path = directory / "models" / model_version / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_promote_blocks_regression_without_force(corpus_root: Path) -> None:
    _write_manifest_with_mae(corpus_root, "mood-2026.09.0", mae=0.2)
    _write_manifest_with_mae(corpus_root, "mood-2026.09.1", mae=0.5)
    (corpus_root / "models" / "active.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "promoted_at": "2026-01-01T00:00:00.000Z"}),
        encoding="utf-8",
    )

    with pytest.raises(MetricRegressionError) as exc:
        promote_model("mood-2026.09.1", models_dir=corpus_root / "models", force=False)
    assert exc.value.code == "PM_R03_METRIC_REGRESSION_BLOCKED"


def test_promote_proceeds_with_force_despite_regression(corpus_root: Path) -> None:
    _write_manifest_with_mae(corpus_root, "mood-2026.09.0", mae=0.2)
    _write_manifest_with_mae(corpus_root, "mood-2026.09.1", mae=0.5)
    (corpus_root / "models" / "active.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "promoted_at": "2026-01-01T00:00:00.000Z"}),
        encoding="utf-8",
    )

    result = promote_model("mood-2026.09.1", models_dir=corpus_root / "models", force=True)
    assert result.regression is True
    assert result.already_active is False


def test_promote_allows_improvement_without_force(corpus_root: Path) -> None:
    _write_manifest_with_mae(corpus_root, "mood-2026.09.0", mae=0.5)
    _write_manifest_with_mae(corpus_root, "mood-2026.09.1", mae=0.2)
    (corpus_root / "models" / "active.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "promoted_at": "2026-01-01T00:00:00.000Z"}),
        encoding="utf-8",
    )

    result = promote_model("mood-2026.09.1", models_dir=corpus_root / "models", force=False)
    assert result.regression is False


def test_promote_is_idempotent_when_already_active(corpus_root: Path) -> None:
    _write_manifest_with_mae(corpus_root, "mood-2026.09.0", mae=0.3)
    (corpus_root / "models" / "active.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "promoted_at": "2026-01-01T00:00:00.000Z"}),
        encoding="utf-8",
    )

    result = promote_model("mood-2026.09.0", models_dir=corpus_root / "models", force=False)
    assert result.already_active is True
