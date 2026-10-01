"""evaluate.metrics: abordagem C — spec 11 Fase 9 (FT-R20, FT-R21, CA-10).

Roda só com requirements-embeddings.txt instalado (marker `embeddings`).
`load_base_encoder` é substituído pelo BERT minúsculo de teste (fixture
`mock_base_encoder`, conftest.py).
"""

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from evaluate.metrics import EvaluateGateError, evaluate_candidate
from tests.conftest import (
    build_synthetic_dataset,
    embeddings_ft_config,
    train_and_stage,
)

pytestmark = pytest.mark.embeddings

_FAST_FINETUNE = {
    "epochs_max": 1,
    "patience": 1,
    "batch_size": 4,
    "lr_encoder": 1e-4,
    "lr_head": 1e-3,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
    "freeze_word_embeddings": True,
}


def _fast_config(train_config: dict) -> dict:
    return embeddings_ft_config(train_config, finetune=_FAST_FINETUNE)


def test_eval_includes_history30_latency_and_omits_comparison_without_a(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    build_synthetic_dataset(corpus_root, "ds-eval-ft-solo")
    config = _fast_config(train_config)
    train_and_stage(corpus_root, "ds-eval-ft-solo", config)

    result = evaluate_candidate(
        "ds-eval-ft-solo", config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets"
    )

    history30 = result.eval_json["latency_history_30"]
    assert history30["sample_size"] > 0
    assert history30["p95_ms"] >= 0.0
    assert "comparison" not in result.eval_json


def test_eval_includes_comparison_when_tfidf_ridge_eval_exists(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    """CA-10: eval.json de C traz comparison.mae_delta_vs_tfidf quando existe
    o de A no mesmo dataset_id."""
    build_synthetic_dataset(corpus_root, "ds-eval-ft-vs-a")

    a_config = {**train_config, "train": {**train_config["train"]}}
    train_and_stage(corpus_root, "ds-eval-ft-vs-a", a_config)
    a_eval = evaluate_candidate(
        "ds-eval-ft-vs-a", a_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets"
    )
    from evaluate.metrics import _write_json_atomic

    _write_json_atomic(a_eval.eval_json, a_eval.staging_dir / "eval.json", "test")

    c_config = _fast_config(train_config)
    train_and_stage(corpus_root, "ds-eval-ft-vs-a", c_config)
    c_result = evaluate_candidate(
        "ds-eval-ft-vs-a", c_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets"
    )

    expected_delta = c_result.eval_json["metrics"]["candidate"]["mae"] - a_eval.eval_json["metrics"]["candidate"]["mae"]
    assert c_result.eval_json["comparison"]["mae_delta_vs_tfidf"] == pytest.approx(expected_delta)


def test_eval_gate_blocked_when_encoder_missing(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    build_synthetic_dataset(corpus_root, "ds-eval-ft-noencoder")
    config = _fast_config(train_config)
    staging_dir = train_and_stage(corpus_root, "ds-eval-ft-noencoder", config)
    shutil.rmtree(staging_dir / "encoder")

    with pytest.raises(EvaluateGateError) as exc:
        evaluate_candidate(
            "ds-eval-ft-noencoder", config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets"
        )
    assert exc.value.code == "EV_R01_GATE_STAGING_NOT_OK"
