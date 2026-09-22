import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

BASE_ROW = {
    "corpus_id": "syn-2026-01-01-a",
    "conversation_id": "syn-conv-0001",
    "customer_id": "syn-cust-0001",
    "message_id": "syn-msg-000001",
    "role": "customer",
    "text": "Qual o horario do check-out?",
    "sent_at": "2026-01-01T10:00:00Z",
    "persona": "objetivo",
    "generated_label": 0.0,
}


def make_row(**overrides: object) -> dict:
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def write_corpus(directory: Path, corpus_id: str, rows: list[dict]) -> Path:
    """Writes `rows` as a .jsonl corpus under <directory>/raw/synthetic/<corpus_id>.jsonl,
    satisfying IG-R02 (path must be under data/raw/synthetic/)."""
    target_dir = directory / "raw" / "synthetic"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{corpus_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def approve_corpus(directory: Path, corpus_id: str, rows: list[dict], *, skip_composition: bool = True) -> Path:
    """Writes a corpus and runs the real validate_corpus() over it, so
    downstream tests (labels, split) start from a genuine, schema-correct
    validation_report.json instead of a hand-forged one. Returns the corpus
    path. Raises if the corpus is not approved (status != "ok")."""
    from ingest.validate import _write_report_atomic, validate_corpus

    path = write_corpus(directory, corpus_id, rows)
    report = validate_corpus(path, skip_composition=skip_composition)
    if report.status != "ok":
        raise AssertionError(f"approve_corpus: corpus was rejected: {report.errors}")
    report_path = directory / "reports" / corpus_id / "validation_report.json"
    _write_report_atomic(report.to_dict(), report_path)
    return path


def approved_sample_corpus(directory: Path, corpus_id: str = "syn-2026-01-01-a") -> Path:
    """Copies the committed sample.jsonl fixture (56 lines, 6 real conversations,
    already known to satisfy DC-R14/R15/R16) into <directory>/raw/synthetic/ and
    runs the real validator, producing a genuine validation_report.json. Used by
    labels/split tests that need a realistic corpus without composition minimums
    (--skip-composition covers only DC-R09..R11, not R14/R15/R16 — a hand-rolled
    1-2 message corpus fails those and can't be used here)."""
    sample_path = Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
    rows = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["corpus_id"] = corpus_id
    return approve_corpus(directory, corpus_id, rows, skip_composition=True)


def approved_medium_corpus(directory: Path, corpus_id: str = "syn-2026-01-02-a") -> Path:
    """Like approved_sample_corpus, but backed by tests/fixtures/sample_medium.jsonl
    (40 customers, 463 lines) — split.py's GroupShuffleSplit needs enough groups
    for train/validation/test to all come out non-empty, which the 6-customer
    sample.jsonl is too small to guarantee."""
    medium_path = Path(__file__).resolve().parent / "fixtures" / "sample_medium.jsonl"
    rows = [json.loads(line) for line in medium_path.read_text(encoding="utf-8").splitlines() if line]
    for row in rows:
        row["corpus_id"] = corpus_id
    return approve_corpus(directory, corpus_id, rows, skip_composition=True)


@pytest.fixture
def medium_jsonl_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sample_medium.jsonl"


def approve_labels(directory: Path, corpus_path: Path) -> Path:
    """Runs the real build_labels() + writes the label set and label_report.json,
    so split tests start from a genuine label gate. Returns the .parquet path."""
    from ingest.validate import sha256_and_size
    from labels.build import _write_parquet_atomic, build_labels
    from labels.build import _write_report_atomic as _write_label_report

    report_dir = directory / "reports"
    labels_dir = directory / "labels"
    rows, meta = build_labels(corpus_path, report_dir)

    parquet_path = labels_dir / f"{meta['label_set_id']}.parquet"
    _write_parquet_atomic(rows, parquet_path, run_id="test")
    parquet_sha256, _ = sha256_and_size(parquet_path)

    report = {
        "report_version": "lb-1",
        "run": {"run_id": "test", "tool": "labels.build"},
        "source": {
            "corpus_id": meta["corpus_id"],
            "corpus_path": meta["corpus_path"],
            "corpus_sha256": meta["corpus_sha256"],
            "ingest_report_path": meta["ingest_report_path"],
        },
        "label_set": {
            "label_set_id": meta["label_set_id"],
            "path": str(parquet_path),
            "sha256": parquet_sha256,
            "rows": len(rows),
        },
        "counts": {
            "messages_read": meta["messages_read"],
            "messages_customer": meta["messages_customer"],
            "messages_agent_excluded": meta["messages_agent_excluded"],
        },
        "distributions": {"label_levels": {}},
    }
    report_path = report_dir / meta["corpus_id"] / "label_report.json"
    _write_label_report(report, report_path, run_id="test")
    return parquet_path


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_jsonl_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"


# ---------------------------------------------------------------------------
# Fase 3 (train/evaluate/registry) — fixtures sintéticos de data/datasets/*.
# train.train's own gate (TN-R01) only checks that dataset.json + the three
# parquet files exist; it never re-verifies the corpus/label hash chain. That
# lets these tests build data/datasets/<dataset_id>/ directly, without paying
# for a real ingest -> labels -> split run.
# ---------------------------------------------------------------------------

_LEVEL_PHRASES = {
    -1.0: "pessimo horrivel absurdo",
    -0.5: "ruim chato demorado",
    0.0: "neutro normal padrao",
    0.5: "bom legal gostei",
    1.0: "otimo excelente maravilhoso",
}
_FILLER_WORDS = ["banana", "carro", "mesa", "livro", "janela", "computador", "caneta", "sapato"]
_PERSONAS = ["objetivo", "cordial", "ansioso", "exigente"]


def build_synthetic_dataset(
    directory: Path,
    dataset_id: str,
    signal: str = "strong",
    n_train: int = 30,
    n_validation: int = 8,
    n_test: int = 8,
) -> Path:
    """Writes train/validation/test.parquet + dataset.json in the exact
    schema of spec 05 §3.2, directly (no ingest/labels/split involved).

    signal="strong": text_clean encodes the label via a distinctive, shared
    phrase per level, so Ridge trivially beats the mean baseline (CA-04, gate
    approved). signal="none": text_clean is random filler unrelated to
    label_score, so the candidate can't beat the baseline (CA-05, gate
    reproved)."""
    rng = random.Random(42)
    levels = sorted(_LEVEL_PHRASES)

    def _make_rows(n: int, split: str, offset: int) -> list[dict]:
        rows = []
        for i in range(n):
            label = levels[i % len(levels)]
            if signal == "strong":
                text = f"{_LEVEL_PHRASES[label]} {' '.join(rng.choices(_FILLER_WORDS, k=2))}"
            else:
                text = " ".join(rng.choices(_FILLER_WORDS, k=5))
            # About half the rows carry non-empty context, like a real corpus
            # (first message of a conversation has none, later ones do) — an
            # all-empty context column leaves the context TfidfVectorizer with
            # no vocabulary to fit.
            context = [f"contexto anterior {i}"] if i % 2 == 0 else []
            rows.append(
                {
                    "example_id": f"syn-msg-{offset + i:06d}",
                    "customer_id": f"syn-cust-{offset + i:04d}",
                    "conversation_id": f"syn-conv-{offset + i:04d}",
                    "persona": _PERSONAS[i % len(_PERSONAS)],
                    "text_clean": text,
                    "context_clean": context,
                    "label_score": label,
                    "split": split,
                }
            )
        return rows

    dataset_dir = directory / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_rows = _make_rows(n_train, "train", 0)
    validation_rows = _make_rows(n_validation, "validation", n_train)
    test_rows = _make_rows(n_test, "test", n_train + n_validation)

    pd.DataFrame(train_rows).to_parquet(dataset_dir / "train.parquet", index=False)
    pd.DataFrame(validation_rows).to_parquet(dataset_dir / "validation.parquet", index=False)
    pd.DataFrame(test_rows).to_parquet(dataset_dir / "test.parquet", index=False)

    dataset_json = {
        "dataset_id": dataset_id,
        "source_ids": {"corpus_id": f"syn-{dataset_id}"},
        "label_set_id": f"ls-syn-{dataset_id}",
        "feature_spec_version": "fs-1",
        "split_strategy": {
            "algorithm": "GroupShuffleSplit",
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "seed": 42,
        },
        "row_counts": {
            "train": {"rows": len(train_rows), "customers": len(train_rows)},
            "validation": {"rows": len(validation_rows), "customers": len(validation_rows)},
            "test": {"rows": len(test_rows), "customers": len(test_rows)},
        },
        "fingerprint": "synthetic-fixture",
        "created_at": "2026-01-01T00:00:00.000Z",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    return dataset_dir


def train_and_stage(directory: Path, dataset_id: str, config: dict) -> Path:
    """Calls the real build_candidate() and writes the staging artifacts —
    same pattern as approve_corpus/approve_labels: exercise the real
    function, not a hand-forged fixture."""
    from train.train import _dump_joblib_atomic, build_candidate
    from train.train import _write_json_atomic as _write_train_manifest

    result = build_candidate(
        dataset_id, config, datasets_dir=directory / "datasets", staging_dir=directory / "staging"
    )
    if not result.reused:
        _dump_joblib_atomic(result.baseline, result.staging_dir / "baseline.joblib", "test")
        _dump_joblib_atomic(result.candidate, result.staging_dir / "candidate.joblib", "test")
        manifest = {**result.train_manifest, "trained_at": "2026-01-01T00:00:00.000Z", "duration_ms": 1}
        _write_train_manifest(manifest, result.staging_dir / "train_manifest.json", "test")
    return result.staging_dir


def evaluate_staging(directory: Path, dataset_id: str, config: dict) -> dict:
    """Calls the real evaluate_candidate() and writes eval.json."""
    from evaluate.metrics import _write_json_atomic as _write_eval_json
    from evaluate.metrics import evaluate_candidate

    result = evaluate_candidate(
        dataset_id, config, staging_dir=directory / "staging", datasets_dir=directory / "datasets"
    )
    eval_json = {**result.eval_json, "evaluated_at": "2026-01-01T00:00:00.000Z", "duration_ms": 1}
    _write_eval_json(eval_json, result.staging_dir / "eval.json", "test")
    return eval_json


@pytest.fixture
def train_config() -> dict:
    return {
        "train": {
            "algorithm": "tfidf-ridge",
            "seed": 42,
            "ridge": {"alpha": 1.0, "solver": "lsqr"},
            "tfidf_word": {"ngram_range": [1, 2]},
            "tfidf_char": {"ngram_range": [2, 5], "analyzer": "char_wb"},
            "context": {"ngram_range": [1, 2], "weight": 0.5},
        },
        "evaluate": {"quality_gate": {"max_mae_ratio": 0.9}},
    }
