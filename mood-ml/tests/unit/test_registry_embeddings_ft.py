"""registry.registry: abordagem C — spec 11 Fase 10 (RG-R18, RG-R19, CA-12).

Roda só com requirements-embeddings.txt instalado (marker `embeddings`).
`load_base_encoder` é substituído pelo BERT minúsculo de teste (fixture
`mock_base_encoder`, conftest.py).
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.conftest import embeddings_ft_config, register_active_model

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
    """epochs_max=1 num BERT aleatório raramente bate a baseline (nada a ver
    com o mecanismo de registro que este arquivo testa) — gate bem permissivo
    para isolar o que de fato está sob teste aqui."""
    config = embeddings_ft_config(train_config, finetune=_FAST_FINETUNE)
    config["evaluate"] = {"quality_gate": {"max_mae_ratio": 10.0}}
    return config


def test_register_copies_encoder_and_extends_manifest(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    config = _fast_config(train_config)
    model_version = register_active_model(corpus_root, "ds-reg-ft", config)

    model_dir = corpus_root / "models" / model_version
    assert (model_dir / "encoder" / "config.json").exists()
    assert (model_dir / "encoder" / "tokenizer_config.json").exists()

    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "embeddings-ft"
    assert set(manifest["encoder"]) == {"source", "revision", "max_length"}
    assert manifest["encoder"]["revision"] == config["train"]["embeddings"]["encoder"]["revision"]


def test_registered_model_predicts_without_the_staging_dir(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    """CA-12: model_version resultante carrega e prediz sem o staging."""
    config = _fast_config(train_config)
    model_version = register_active_model(corpus_root, "ds-reg-ft-standalone", config)
    model_dir = corpus_root / "models" / model_version

    shutil.rmtree(corpus_root / "staging")

    candidate = joblib.load(model_dir / "model.joblib")
    candidate.attach(model_dir / "encoder")
    df = pd.DataFrame({"text_clean": ["oi tudo bem"], "context_clean": [[]]})
    preds = candidate.predict(df)

    assert preds.shape == (1,)
    assert np.isfinite(preds[0])
