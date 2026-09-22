import json
import subprocess
import sys
from pathlib import Path

import yaml

import evaluate.metrics as evaluate_cli
import train.train as train_cli
from tests.conftest import build_synthetic_dataset, train_and_stage

MOOD_ML_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_config(directory: Path, config: dict) -> Path:
    path = directory / "pipeline.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(directory: Path, dataset_id: str, config_path: Path) -> int:
    return evaluate_cli.main(
        [
            dataset_id,
            "--config",
            str(config_path),
            "--datasets-dir",
            str(directory / "datasets"),
            "--staging-dir",
            str(directory / "staging"),
        ]
    )


def test_exit_0_and_eval_json_written_when_gate_passes(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-a", signal="strong")
    config_path = _write_config(corpus_root, train_config)
    assert train_cli.main(
        [
            "ds-a",
            "--config",
            str(config_path),
            "--datasets-dir",
            str(corpus_root / "datasets"),
            "--staging-dir",
            str(corpus_root / "staging"),
        ]
    ) == 0

    code = _run(corpus_root, "ds-a", config_path)
    assert code == 0

    staging = next((corpus_root / "staging" / "ds-a" / "tfidf-ridge").glob("*"))
    eval_json = json.loads((staging / "eval.json").read_text(encoding="utf-8"))
    assert eval_json["gate"]["passed"] is True
    assert eval_json["dataset_id"] == "ds-a"


def test_exit_5_when_gate_fails_but_eval_json_is_still_written(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-b", signal="none")
    config_path = _write_config(corpus_root, train_config)
    assert train_cli.main(
        [
            "ds-b",
            "--config",
            str(config_path),
            "--datasets-dir",
            str(corpus_root / "datasets"),
            "--staging-dir",
            str(corpus_root / "staging"),
        ]
    ) == 0

    code = _run(corpus_root, "ds-b", config_path)
    assert code == 5

    staging = next((corpus_root / "staging" / "ds-b" / "tfidf-ridge").glob("*"))
    eval_json_path = staging / "eval.json"
    assert eval_json_path.exists()
    eval_json = json.loads(eval_json_path.read_text(encoding="utf-8"))
    assert eval_json["gate"]["passed"] is False


def test_exit_3_when_staging_is_not_ok(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-c")
    config_path = _write_config(corpus_root, train_config)

    # No train run: staging/ was never created.
    code = _run(corpus_root, "ds-c", config_path)
    assert code == 3
    assert not (corpus_root / "staging" / "ds-c" / "tfidf-ridge").exists()


def test_rerun_produces_the_same_gate_outcome(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-d", signal="strong")
    config_path = _write_config(corpus_root, train_config)
    train_and_stage(corpus_root, "ds-d", train_config)

    code1 = _run(corpus_root, "ds-d", config_path)
    staging = next((corpus_root / "staging" / "ds-d" / "tfidf-ridge").glob("*"))
    eval_json_1 = json.loads((staging / "eval.json").read_text(encoding="utf-8"))

    code2 = _run(corpus_root, "ds-d", config_path)
    eval_json_2 = json.loads((staging / "eval.json").read_text(encoding="utf-8"))

    assert code1 == code2
    assert eval_json_1["metrics"] == eval_json_2["metrics"]


def test_no_duckdb_reference_in_evaluate() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "evaluate" / "metrics.py"
    assert "duckdb" not in module_path.read_text(encoding="utf-8").lower()


def test_candidate_trained_as_real_main_entrypoint_loads_in_a_separate_process(corpus_root: Path, train_config: dict) -> None:
    """Regression test: `python -m train.train` sets train.train's __name__
    to "__main__", so a plain `def join_context_list(...)` inside it gets
    pickled by joblib with module="__main__". Unpickling later from
    `python -m evaluate.metrics` (a different real entry point) then fails,
    because THAT process's own __main__ is evaluate.metrics, not train.train.
    In-process calls to train.main()/evaluate_candidate() never hit this
    because train.train's __name__ stays "train.train" when imported —
    only real subprocess invocations reproduce it."""
    build_synthetic_dataset(corpus_root, "ds-proc", signal="strong")
    config_path = _write_config(corpus_root, train_config)
    common_args = [
        "ds-proc",
        "--config",
        str(config_path),
        "--datasets-dir",
        str(corpus_root / "datasets"),
        "--staging-dir",
        str(corpus_root / "staging"),
    ]

    train_proc = subprocess.run(
        [sys.executable, "-m", "train.train", *common_args],
        cwd=MOOD_ML_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert train_proc.returncode == 0, train_proc.stderr

    evaluate_proc = subprocess.run(
        [sys.executable, "-m", "evaluate.metrics", *common_args],
        cwd=MOOD_ML_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert evaluate_proc.returncode == 0, evaluate_proc.stderr
