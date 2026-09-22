import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate.metrics import compute_training_fingerprint
from tests.conftest import build_synthetic_dataset
from train.train import (
    TrainGateError,
    TrainSanityError,
    _build_pipeline,
    build_candidate,
    join_context_list,
)

# ---------------------------------------------------------------------------
# join_context_list
# ---------------------------------------------------------------------------


def test_join_context_list_handles_python_lists_and_empty() -> None:
    column = [["oi", "tudo bem"], [], ["um item"]]
    assert join_context_list(column) == ["oi tudo bem", "", "um item"]


def test_join_context_list_handles_numpy_arrays_from_parquet_roundtrip() -> None:
    """A Parquet round-trip through pyarrow turns list columns into numpy
    arrays, not plain Python lists — `if item` on a multi-element array
    raises ValueError, which is exactly what this guards against."""
    column = [np.array(["oi", "tudo"]), np.array([], dtype=object), np.array(["x"])]
    assert join_context_list(column) == ["oi tudo", "", "x"]


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


def test_training_fingerprint_is_deterministic() -> None:
    dataset_sha256 = {"train": "a", "validation": "b", "test": "c"}
    hp = {"seed": 42, "ridge": {"alpha": 1.0}}
    fp1 = compute_training_fingerprint(dataset_sha256, "tfidf-ridge", hp, "fs-1")
    fp2 = compute_training_fingerprint(dataset_sha256, "tfidf-ridge", hp, "fs-1")
    assert fp1 == fp2


def test_training_fingerprint_changes_with_hyperparameters() -> None:
    dataset_sha256 = {"train": "a", "validation": "b", "test": "c"}
    fp1 = compute_training_fingerprint(dataset_sha256, "tfidf-ridge", {"ridge": {"alpha": 1.0}}, "fs-1")
    fp2 = compute_training_fingerprint(dataset_sha256, "tfidf-ridge", {"ridge": {"alpha": 2.0}}, "fs-1")
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# build_candidate — gate, idempotency, sanity
# ---------------------------------------------------------------------------


def test_gate_blocked_when_dataset_json_missing(corpus_root: Path, train_config: dict) -> None:
    with pytest.raises(TrainGateError) as exc:
        build_candidate("ds-missing", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    assert exc.value.code == "TN_R01_GATE_DATASET_NOT_OK"


def test_gate_blocked_when_a_parquet_is_missing(corpus_root: Path, train_config: dict) -> None:
    dataset_dir = build_synthetic_dataset(corpus_root, "ds-broken")
    (dataset_dir / "test.parquet").unlink()

    with pytest.raises(TrainGateError) as exc:
        build_candidate("ds-broken", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    assert exc.value.code == "TN_R01_GATE_DATASET_NOT_OK"


def test_no_op_when_staging_already_complete(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-a")
    result1 = build_candidate("ds-a", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    assert result1.reused is False

    # Write the staging artifacts for real (mirrors what the CLI does), then
    # rerun build_candidate and confirm it reuses instead of refitting.
    from train.train import _dump_joblib_atomic
    from train.train import _write_json_atomic as _write_manifest

    _dump_joblib_atomic(result1.baseline, result1.staging_dir / "baseline.joblib", "test")
    _dump_joblib_atomic(result1.candidate, result1.staging_dir / "candidate.joblib", "test")
    _write_manifest(
        {**result1.train_manifest, "trained_at": "2026-01-01T00:00:00.000Z", "duration_ms": 1},
        result1.staging_dir / "train_manifest.json",
        "test",
    )

    result2 = build_candidate("ds-a", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    assert result2.reused is True
    assert result2.train_manifest["fingerprint"] == result1.train_manifest["fingerprint"]


def test_different_hyperparameters_produce_different_staging_dir(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-b")
    result1 = build_candidate("ds-b", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")

    other_config = {**train_config, "train": {**train_config["train"], "ridge": {"alpha": 5.0, "solver": "lsqr"}}}
    result2 = build_candidate("ds-b", other_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")

    assert result1.staging_dir != result2.staging_dir


def test_unsupported_algorithm_raises_value_error(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-c")
    bad_config = {**train_config, "train": {**train_config["train"], "algorithm": "embeddings-ridge"}}
    with pytest.raises(ValueError):
        build_candidate("ds-c", bad_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")


def test_sanity_check_catches_non_finite_prediction(
    corpus_root: Path, train_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import train.train as train_mod

    build_synthetic_dataset(corpus_root, "ds-d")
    monkeypatch.setattr(train_mod.Ridge, "predict", lambda self, X: np.full(X.shape[0], np.nan))

    with pytest.raises(TrainSanityError) as exc:
        build_candidate("ds-d", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    assert exc.value.code == "TN_R14_NON_FINITE_PREDICTION"


def test_validation_diagnostic_is_computed_and_finite(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-e")
    result = build_candidate("ds-e", train_config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging")
    mae = result.train_manifest["validation_mae_candidate"]
    assert np.isfinite(mae)
    assert mae >= 0.0


# ---------------------------------------------------------------------------
# TN-R16 — vocabulary must never carry unmasked PII (CA-09)
# ---------------------------------------------------------------------------


def _vocabularies(pipeline) -> tuple[set[str], set[str]]:
    features = pipeline.named_steps["features"]
    word_vec = features.named_transformers_["text"].transformer_list[0][1]
    char_vec = features.named_transformers_["text"].transformer_list[1][1]
    return set(word_vec.vocabulary_.keys()), set(char_vec.vocabulary_.keys())


_CPF_FRAGMENT_RE = re.compile(r"\d{2,3}[.\-]\d")


def _has_pii_shaped_token(char_vocab: set[str]) -> bool:
    return any("@" in token or _CPF_FRAGMENT_RE.search(token) for token in char_vocab)


def _fit_on_rows(rows: list[dict], hp: dict):
    df = pd.DataFrame(rows)
    pipeline = _build_pipeline(hp)
    pipeline.fit(df, df["label_score"].to_numpy())
    return pipeline


def test_vocabulary_leak_test_actually_detects_unmasked_pii(train_config: dict) -> None:
    """CA-09: proves the detection mechanism works, by deliberately feeding
    unmasked PII (simulating a T2 regression) and confirming it's caught."""
    rows = [
        {"text_clean": "meu email vazado eh joaosilva@example.com aqui", "context_clean": ["contexto"], "label_score": 1.0},
        {"text_clean": "cpf vazado 123.456.789-01 aqui tambem", "context_clean": [], "label_score": -1.0},
        {"text_clean": "texto neutro qualquer sem nada de especial", "context_clean": [], "label_score": 0.0},
        {"text_clean": "mais um texto de exemplo variado", "context_clean": ["outro contexto"], "label_score": 0.5},
    ]
    pipeline = _fit_on_rows(rows, train_config["train"])
    _word_vocab, char_vocab = _vocabularies(pipeline)
    assert _has_pii_shaped_token(char_vocab), "the leak test should have caught the deliberately unmasked PII"


def test_vocabulary_has_no_pii_shaped_token_when_properly_masked(train_config: dict) -> None:
    """The actual TN-R16 regression guard: text_clean as T2 would really
    produce (markers, never raw PII) leaves no PII-shaped trace."""
    rows = [
        {"text_clean": "meu email e <EMAIL>, obrigado", "context_clean": ["contexto"], "label_score": 1.0},
        {"text_clean": "meu cpf e <CPF> por favor confirme", "context_clean": [], "label_score": -1.0},
        {"text_clean": "pode me ligar no <TEL> amanha", "context_clean": [], "label_score": 0.0},
        {"text_clean": "texto qualquer sem nenhum dado pessoal", "context_clean": ["outro"], "label_score": 0.5},
    ]
    pipeline = _fit_on_rows(rows, train_config["train"])
    _word_vocab, char_vocab = _vocabularies(pipeline)
    assert not _has_pii_shaped_token(char_vocab)
