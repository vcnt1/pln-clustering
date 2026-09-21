import json
from pathlib import Path

import transform.split as split_cli
from tests.conftest import approve_labels, approved_medium_corpus, write_corpus


def test_exit_0_and_artifacts_written(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)

    code = split_cli.main(
        [
            str(path),
            "--dataset-id",
            "ds-2026-01-02-a",
            "--report-dir",
            str(corpus_root / "reports"),
            "--datasets-dir",
            str(corpus_root / "datasets"),
        ]
    )
    assert code == 0

    dataset_dir = corpus_root / "datasets" / "ds-2026-01-02-a"
    assert (dataset_dir / "train.parquet").exists()
    assert (dataset_dir / "validation.parquet").exists()
    assert (dataset_dir / "test.parquet").exists()
    dataset_json = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    assert dataset_json["dataset_id"] == "ds-2026-01-02-a"
    assert dataset_json["label_set_id"] == "ls-syn-2026-01-02-a"
    assert "created_at" in dataset_json


def test_exit_3_when_corpus_not_approved(corpus_root: Path, medium_jsonl_path: Path) -> None:
    rows = [json.loads(line) for line in medium_jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    path = write_corpus(corpus_root, "syn-2026-01-02-a", rows)

    code = split_cli.main(
        [
            str(path),
            "--dataset-id",
            "ds-2026-01-02-a",
            "--report-dir",
            str(corpus_root / "reports"),
            "--datasets-dir",
            str(corpus_root / "datasets"),
        ]
    )
    assert code == 3
    assert not (corpus_root / "datasets").exists()


def test_rerun_with_same_inputs_is_byte_identical(corpus_root: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)
    args = [
        str(path),
        "--dataset-id",
        "ds-2026-01-02-a",
        "--report-dir",
        str(corpus_root / "reports"),
        "--datasets-dir",
        str(corpus_root / "datasets"),
    ]

    assert split_cli.main(args) == 0
    dataset_dir = corpus_root / "datasets" / "ds-2026-01-02-a"
    train_bytes_1 = (dataset_dir / "train.parquet").read_bytes()
    test_bytes_1 = (dataset_dir / "test.parquet").read_bytes()

    assert split_cli.main(args) == 0
    train_bytes_2 = (dataset_dir / "train.parquet").read_bytes()
    test_bytes_2 = (dataset_dir / "test.parquet").read_bytes()

    assert train_bytes_1 == train_bytes_2
    assert test_bytes_1 == test_bytes_2


def test_no_duckdb_reference_in_split() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "transform" / "split.py"
    text = module_path.read_text(encoding="utf-8")
    assert "duckdb" not in text.lower()


def test_no_raw_text_in_dataset_json(corpus_root: Path, medium_jsonl_path: Path) -> None:
    path = approved_medium_corpus(corpus_root)
    approve_labels(corpus_root, path)
    split_cli.main(
        [
            str(path),
            "--dataset-id",
            "ds-2026-01-02-a",
            "--report-dir",
            str(corpus_root / "reports"),
            "--datasets-dir",
            str(corpus_root / "datasets"),
        ]
    )
    dataset_json_text = (corpus_root / "datasets" / "ds-2026-01-02-a" / "dataset.json").read_text(encoding="utf-8")

    raw_rows = [json.loads(line) for line in medium_jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    raw_texts = {row["text"] for row in raw_rows}
    for text in raw_texts:
        assert text not in dataset_json_text
