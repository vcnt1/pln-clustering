import json
from pathlib import Path

import yaml

import train.train as train_cli
from tests.conftest import build_synthetic_dataset


def _write_config(directory: Path, config: dict) -> Path:
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_exit_0_and_artifacts_written(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-a")
    config_path = _write_config(corpus_root, train_config)

    code = train_cli.main(
        [
            "ds-a",
            "--config",
            str(config_path),
            "--datasets-dir",
            str(corpus_root / "datasets"),
            "--staging-dir",
            str(corpus_root / "staging"),
        ]
    )
    assert code == 0

    staging_dirs = list((corpus_root / "staging" / "ds-a" / "tfidf-ridge").glob("*"))
    assert len(staging_dirs) == 1
    staging = staging_dirs[0]
    assert (staging / "baseline.joblib").exists()
    assert (staging / "candidate.joblib").exists()
    manifest = json.loads((staging / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "ds-a"
    assert manifest["algorithm"] == "tfidf-ridge"
    assert "trained_at" in manifest


def test_exit_3_when_dataset_not_approved(corpus_root: Path, train_config: dict) -> None:
    config_path = _write_config(corpus_root, train_config)

    code = train_cli.main(
        [
            "ds-missing",
            "--config",
            str(config_path),
            "--datasets-dir",
            str(corpus_root / "datasets"),
            "--staging-dir",
            str(corpus_root / "staging"),
        ]
    )
    assert code == 3
    assert not (corpus_root / "staging").exists()


def _train_argv(corpus_root: Path, dataset_id: str, config_path: Path) -> list[str]:
    return [
        dataset_id,
        "--config",
        str(config_path),
        "--datasets-dir",
        str(corpus_root / "datasets"),
        "--staging-dir",
        str(corpus_root / "staging"),
    ]


def test_exit_3_when_dataset_json_is_corrupted(corpus_root: Path, train_config: dict) -> None:
    dataset_dir = build_synthetic_dataset(corpus_root, "ds-corrupt")
    (dataset_dir / "dataset.json").write_text("{not json", encoding="utf-8")
    config_path = _write_config(corpus_root, train_config)

    assert train_cli.main(_train_argv(corpus_root, "ds-corrupt", config_path)) == 3
    assert not (corpus_root / "staging").exists()


def test_exit_2_when_algorithm_is_unsupported(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-algo")
    bad_config = {**train_config, "train": {**train_config["train"], "algorithm": "embeddings-ridge"}}
    config_path = _write_config(corpus_root, bad_config)

    assert train_cli.main(_train_argv(corpus_root, "ds-algo", config_path)) == 2


def test_exit_1_when_fit_raises_value_error(corpus_root: Path, train_config: dict, monkeypatch) -> None:
    build_synthetic_dataset(corpus_root, "ds-fit")
    config_path = _write_config(corpus_root, train_config)

    def _raise(_hyperparameters: dict) -> None:
        raise ValueError("empty vocabulary; perhaps the documents only contain stop words")

    monkeypatch.setattr(train_cli, "_build_pipeline", _raise)

    assert train_cli.main(_train_argv(corpus_root, "ds-fit", config_path)) == 1


def test_rerun_does_not_rewrite_manifest(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-b")
    config_path = _write_config(corpus_root, train_config)
    args = [
        "ds-b",
        "--config",
        str(config_path),
        "--datasets-dir",
        str(corpus_root / "datasets"),
        "--staging-dir",
        str(corpus_root / "staging"),
    ]

    assert train_cli.main(args) == 0
    staging = next((corpus_root / "staging" / "ds-b" / "tfidf-ridge").glob("*"))
    manifest_path = staging / "train_manifest.json"
    mtime_1 = manifest_path.stat().st_mtime_ns

    assert train_cli.main(args) == 0
    mtime_2 = manifest_path.stat().st_mtime_ns
    assert mtime_1 == mtime_2


def test_different_hyperparameter_creates_new_staging_and_keeps_old(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-c")
    config_path_1 = _write_config(corpus_root, train_config)
    args1 = [
        "ds-c",
        "--config",
        str(config_path_1),
        "--datasets-dir",
        str(corpus_root / "datasets"),
        "--staging-dir",
        str(corpus_root / "staging"),
    ]
    assert train_cli.main(args1) == 0
    first_staging = next((corpus_root / "staging" / "ds-c" / "tfidf-ridge").glob("*"))
    first_manifest_bytes = (first_staging / "train_manifest.json").read_bytes()

    other_config = {**train_config, "train": {**train_config["train"], "ridge": {"alpha": 7.0, "solver": "lsqr"}}}
    config_path_2 = corpus_root / "pipeline2.yaml"
    config_path_2.write_text(yaml.safe_dump(other_config), encoding="utf-8")
    args2 = [
        "ds-c",
        "--config",
        str(config_path_2),
        "--datasets-dir",
        str(corpus_root / "datasets"),
        "--staging-dir",
        str(corpus_root / "staging"),
    ]
    assert train_cli.main(args2) == 0

    all_staging_dirs = list((corpus_root / "staging" / "ds-c" / "tfidf-ridge").glob("*"))
    assert len(all_staging_dirs) == 2
    # the original staging directory's manifest is untouched
    assert (first_staging / "train_manifest.json").read_bytes() == first_manifest_bytes


def test_no_duckdb_reference_in_train() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "train" / "train.py"
    assert "duckdb" not in module_path.read_text(encoding="utf-8").lower()
