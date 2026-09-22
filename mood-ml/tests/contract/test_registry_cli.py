import json
from pathlib import Path

import yaml

import evaluate.metrics as evaluate_cli
import registry.registry as registry_cli
import train.train as train_cli
from tests.conftest import build_synthetic_dataset


def _write_config(directory: Path, config: dict) -> Path:
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _prepare_staged_candidate(directory: Path, dataset_id: str, config: dict, signal: str = "strong") -> str:
    """Runs train.train + evaluate.metrics through their real CLIs and
    returns the resulting fingerprint (read back from eval.json)."""
    build_synthetic_dataset(directory, dataset_id, signal=signal)
    config_path = _write_config(directory, config)
    common = [
        "--config",
        str(config_path),
        "--datasets-dir",
        str(directory / "datasets"),
        "--staging-dir",
        str(directory / "staging"),
    ]
    assert train_cli.main([dataset_id, *common]) == 0
    evaluate_cli.main([dataset_id, *common])  # exit code varies with signal; eval.json is what matters here

    staging = next((directory / "staging" / dataset_id / "tfidf-ridge").glob("*"))
    eval_json = json.loads((staging / "eval.json").read_text(encoding="utf-8"))
    return eval_json["fingerprint"]


def _register_args(directory: Path, dataset_id: str, fingerprint: str) -> list[str]:
    return [
        "register",
        dataset_id,
        fingerprint,
        "--staging-dir",
        str(directory / "staging"),
        "--models-dir",
        str(directory / "models"),
        "--datasets-dir",
        str(directory / "datasets"),
    ]


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


def test_register_exit_0_writes_model_and_manifest(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _prepare_staged_candidate(corpus_root, "ds-a", train_config, signal="strong")

    code = registry_cli.main(_register_args(corpus_root, "ds-a", fingerprint))
    assert code == 0

    model_dirs = list((corpus_root / "models").glob("mood-*"))
    assert len(model_dirs) == 1
    model_dir = model_dirs[0]
    assert (model_dir / "model.joblib").exists()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_fingerprint"] == fingerprint
    assert manifest["history_window"] == 30
    assert manifest["mood_labels"] is None


def test_register_rerun_is_idempotent_no_new_version(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _prepare_staged_candidate(corpus_root, "ds-b", train_config, signal="strong")

    assert registry_cli.main(_register_args(corpus_root, "ds-b", fingerprint)) == 0
    first_dirs = sorted(d.name for d in (corpus_root / "models").glob("mood-*"))

    assert registry_cli.main(_register_args(corpus_root, "ds-b", fingerprint)) == 0
    second_dirs = sorted(d.name for d in (corpus_root / "models").glob("mood-*"))

    assert first_dirs == second_dirs
    assert len(second_dirs) == 1


def test_register_exit_3_when_staging_missing(corpus_root: Path) -> None:
    code = registry_cli.main(_register_args(corpus_root, "ds-missing", "0" * 16))
    assert code == 3
    assert not (corpus_root / "models").exists()


def test_register_exit_6_when_gate_failed(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _prepare_staged_candidate(corpus_root, "ds-c", train_config, signal="none")

    code = registry_cli.main(_register_args(corpus_root, "ds-c", fingerprint))
    assert code == 6
    assert not (corpus_root / "models").exists()


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_exit_0_writes_active_json(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _prepare_staged_candidate(corpus_root, "ds-d", train_config, signal="strong")
    assert registry_cli.main(_register_args(corpus_root, "ds-d", fingerprint)) == 0
    model_version = next((corpus_root / "models").glob("mood-*")).name

    code = registry_cli.main(
        ["promote", model_version, "--models-dir", str(corpus_root / "models")]
    )
    assert code == 0

    active = json.loads((corpus_root / "models" / "active.json").read_text(encoding="utf-8"))
    assert active["model_version"] == model_version


def test_promote_exit_3_when_model_not_found(corpus_root: Path) -> None:
    code = registry_cli.main(
        ["promote", "mood-2099.01.0", "--models-dir", str(corpus_root / "models")]
    )
    assert code == 3


def test_promote_rerun_is_idempotent(corpus_root: Path, train_config: dict) -> None:
    fingerprint = _prepare_staged_candidate(corpus_root, "ds-e", train_config, signal="strong")
    assert registry_cli.main(_register_args(corpus_root, "ds-e", fingerprint)) == 0
    model_version = next((corpus_root / "models").glob("mood-*")).name

    args = ["promote", model_version, "--models-dir", str(corpus_root / "models")]
    assert registry_cli.main(args) == 0
    active_1 = (corpus_root / "models" / "active.json").read_text(encoding="utf-8")

    assert registry_cli.main(args) == 0
    active_2 = (corpus_root / "models" / "active.json").read_text(encoding="utf-8")
    assert json.loads(active_1)["model_version"] == json.loads(active_2)["model_version"]


def test_promote_exit_7_blocks_regression_without_force(corpus_root: Path, train_config: dict) -> None:
    fingerprint_a = _prepare_staged_candidate(corpus_root, "ds-f", train_config, signal="strong")
    assert registry_cli.main(_register_args(corpus_root, "ds-f", fingerprint_a)) == 0
    version_a = next((corpus_root / "models").glob("mood-*")).name
    assert registry_cli.main(["promote", version_a, "--models-dir", str(corpus_root / "models")]) == 0

    # Force a second registered version with a manually-degraded MAE so the
    # comparison in promote_model has something worse to react to.
    other_config = {**train_config, "train": {**train_config["train"], "ridge": {"alpha": 9.0, "solver": "lsqr"}}}
    fingerprint_b = _prepare_staged_candidate(corpus_root, "ds-g", other_config, signal="strong")
    assert registry_cli.main(_register_args(corpus_root, "ds-g", fingerprint_b)) == 0
    version_b = next(d.name for d in (corpus_root / "models").glob("mood-*") if d.name != version_a)

    manifest_path = corpus_root / "models" / version_b / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    active_manifest = json.loads((corpus_root / "models" / version_a / "manifest.json").read_text(encoding="utf-8"))
    manifest["metrics"]["mae"] = active_manifest["metrics"]["mae"] + 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    code = registry_cli.main(["promote", version_b, "--models-dir", str(corpus_root / "models")])
    assert code == 7
    active = json.loads((corpus_root / "models" / "active.json").read_text(encoding="utf-8"))
    assert active["model_version"] == version_a  # unchanged

    code_forced = registry_cli.main(["promote", version_b, "--force", "--models-dir", str(corpus_root / "models")])
    assert code_forced == 0
    active = json.loads((corpus_root / "models" / "active.json").read_text(encoding="utf-8"))
    assert active["model_version"] == version_b


def test_no_duckdb_reference_in_registry() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "registry" / "registry.py"
    assert "duckdb" not in module_path.read_text(encoding="utf-8").lower()
