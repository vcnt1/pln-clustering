import json
from datetime import date
from pathlib import Path

from pipeline import (
    _find_model_version_by_fingerprint,
    _mint_dataset_id,
    _recompute_fingerprint,
)
from tests.conftest import build_synthetic_dataset, train_and_stage

# ---------------------------------------------------------------------------
# _mint_dataset_id — PL-R05
# ---------------------------------------------------------------------------


def test_mint_dataset_id_starts_at_a_when_empty(tmp_path: Path) -> None:
    dataset_id = _mint_dataset_id(tmp_path / "datasets", today=date(2026, 9, 22))
    assert dataset_id == "ds-2026-09-22-a"


def test_mint_dataset_id_increments_letter_when_taken(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    (datasets_dir / "ds-2026-09-22-a").mkdir(parents=True)

    dataset_id = _mint_dataset_id(datasets_dir, today=date(2026, 9, 22))
    assert dataset_id == "ds-2026-09-22-b"


def test_mint_dataset_id_skips_multiple_taken_letters(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    (datasets_dir / "ds-2026-09-22-a").mkdir(parents=True)
    (datasets_dir / "ds-2026-09-22-b").mkdir(parents=True)
    (datasets_dir / "ds-2026-09-22-c").mkdir(parents=True)

    dataset_id = _mint_dataset_id(datasets_dir, today=date(2026, 9, 22))
    assert dataset_id == "ds-2026-09-22-d"


def test_mint_dataset_id_does_not_collide_across_different_days(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    (datasets_dir / "ds-2026-09-21-a").mkdir(parents=True)

    dataset_id = _mint_dataset_id(datasets_dir, today=date(2026, 9, 22))
    assert dataset_id == "ds-2026-09-22-a"


# ---------------------------------------------------------------------------
# _recompute_fingerprint — PL-R09
# ---------------------------------------------------------------------------


def test_recompute_fingerprint_matches_what_train_actually_wrote(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-a", signal="strong")
    staging_dir = train_and_stage(corpus_root, "ds-a", train_config)

    manifest = json.loads((staging_dir / "train_manifest.json").read_text(encoding="utf-8"))
    real_fingerprint = manifest["fingerprint"]

    recomputed = _recompute_fingerprint("ds-a", train_config, corpus_root / "datasets")
    assert recomputed == real_fingerprint


def test_recompute_fingerprint_changes_with_different_hyperparameters(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-b", signal="strong")
    train_and_stage(corpus_root, "ds-b", train_config)

    other_config = {**train_config, "train": {**train_config["train"], "ridge": {"alpha": 9.0, "solver": "lsqr"}}}
    fp1 = _recompute_fingerprint("ds-b", train_config, corpus_root / "datasets")
    fp2 = _recompute_fingerprint("ds-b", other_config, corpus_root / "datasets")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# _find_model_version_by_fingerprint
# ---------------------------------------------------------------------------


def test_find_model_version_by_fingerprint_finds_a_match(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "mood-2026.09.0"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "training_fingerprint": "abc123"}), encoding="utf-8"
    )

    assert _find_model_version_by_fingerprint(models_dir, "abc123") == "mood-2026.09.0"


def test_find_model_version_by_fingerprint_returns_none_without_match(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_dir = models_dir / "mood-2026.09.0"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps({"model_version": "mood-2026.09.0", "training_fingerprint": "abc123"}), encoding="utf-8"
    )

    assert _find_model_version_by_fingerprint(models_dir, "does-not-exist") is None


def test_find_model_version_by_fingerprint_handles_no_models_dir(tmp_path: Path) -> None:
    assert _find_model_version_by_fingerprint(tmp_path / "models", "abc123") is None
