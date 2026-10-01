"""infer.predict / main.py: abordagem C — spec 11 Fase 11 (IF-R23, IF-R24,
CA-07, CA-08).

Roda só com requirements-embeddings.txt instalado (marker `embeddings`).
`load_base_encoder` é substituído pelo BERT minúsculo de teste (fixture
`mock_base_encoder`, conftest.py).
"""

import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from fastapi.testclient import TestClient

from infer.predict import (
    EmbeddingsUnavailableError,
    RegistryBrokenError,
    load_active_model,
    warm_up,
)
from main import create_app
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
    config = embeddings_ft_config(train_config, finetune=_FAST_FINETUNE)
    config["evaluate"] = {"quality_gate": {"max_mae_ratio": 10.0}}
    return config


def test_load_active_model_attaches_encoder_and_warms_up(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    config = _fast_config(train_config)
    register_active_model(corpus_root, "ds-infer-ft", config)

    state = load_active_model(corpus_root / "models")

    assert state.manifest["algorithm"] == "embeddings-ft"
    warm_up(state.pipeline)  # não deve lançar (IF-R24)


def test_startup_refused_when_encoder_missing(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    """CA-07: serviço com modelo C ativo e encoder/ removido não sobe."""
    config = _fast_config(train_config)
    model_version = register_active_model(corpus_root, "ds-infer-ft-noenc", config)
    shutil.rmtree(corpus_root / "models" / model_version / "encoder")

    with pytest.raises(RegistryBrokenError):
        load_active_model(corpus_root / "models")


def test_startup_refused_without_torch(
    corpus_root: Path, train_config: dict, mock_base_encoder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """IF-R23/FT-R18: sem torch instalado, a subida é recusada com FT_R02_DEPS_MISSING."""
    config = _fast_config(train_config)
    register_active_model(corpus_root, "ds-infer-ft-notorch", config)
    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(EmbeddingsUnavailableError) as exc:
        load_active_model(corpus_root / "models")
    assert exc.value.code == "FT_R02_DEPS_MISSING"


def test_service_serves_predictions_fully_offline(
    corpus_root: Path, train_config: dict, mock_base_encoder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CA-08: com a rede bloqueada (HF_HUB_OFFLINE=1), o serviço sobe e prediz (FT-R17)."""
    config = _fast_config(train_config)
    register_active_model(corpus_root, "ds-infer-ft-offline", config)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    app = create_app(models_dir=corpus_root / "models")
    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/infer",
            json={
                "request_id": "req-1",
                "customer_id": "cust-1",
                "conversation_id": "conv-1",
                "history": [{"message_id": "m1", "role": "customer", "text": "oi tudo bem"}],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert -1.0 <= body["score"] <= 1.0
