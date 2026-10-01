"""Abordagem C (spec 11): pooling, codificação e o wrapper EmbeddingMoodModel.

Roda só com requirements-embeddings.txt instalado (pytest.ini: marker
`embeddings`); pulado via importorskip quando torch/transformers não estão
presentes, para que a suíte de A continue verde sem a dependência pesada
(CA-06).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.conftest import build_tiny_encoder
from transform.embeddings import (
    EmbeddingMoodModel,
    EncoderSettings,
    build_features,
    encode_context,
    encode_texts,
    mean_pool,
    resolve_device,
)

pytestmark = pytest.mark.embeddings


# ---------------------------------------------------------------------------
# mean_pool
# ---------------------------------------------------------------------------


def test_mean_pool_ignores_padding() -> None:
    # 2 tokens reais + 1 de padding; o padding tem valores absurdos que não
    # podem influenciar a média.
    tokens = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    pooled = mean_pool(tokens, attention_mask)

    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


# ---------------------------------------------------------------------------
# encode_context — FT-R06 (vetor nulo / média) e FT-R08 (sem gradiente)
# ---------------------------------------------------------------------------


def test_encode_context_empty_list_is_the_null_vector(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)

    result = encode_context(encoder, tokenizer, [[]], max_length=16, device="cpu")

    assert torch.allclose(result, torch.zeros_like(result))


def test_encode_context_matches_manual_mean_for_one_and_many_messages(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    one_message = ["tok1 tok2"]
    many_messages = [f"tok{i} tok{i + 1}" for i in range(29)]

    with torch.no_grad():
        expected_one = encode_texts(encoder, tokenizer, one_message, max_length=16, device="cpu").mean(dim=0)
        expected_many = encode_texts(encoder, tokenizer, many_messages, max_length=16, device="cpu").mean(dim=0)

    result = encode_context(encoder, tokenizer, [one_message, many_messages], max_length=16, device="cpu")

    assert torch.allclose(result[0], expected_one, atol=1e-5)
    assert torch.allclose(result[1], expected_many, atol=1e-5)


def test_encode_context_never_leaves_gradients_on_its_output(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    result = encode_context(encoder, tokenizer, [["tok1 tok2"]], max_length=16, device="cpu")

    assert result.requires_grad is False


def test_encode_context_restores_training_mode(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    encoder.train()

    encode_context(encoder, tokenizer, [["tok1 tok2"]], max_length=16, device="cpu")

    assert encoder.training is True


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


def test_build_features_concatenates_text_and_weighted_context() -> None:
    text = torch.tensor([[1.0, 2.0]])
    context = torch.tensor([[3.0, 4.0]])

    features = build_features(text, context, weight=0.5)

    assert torch.allclose(features, torch.tensor([[1.0, 2.0, 1.5, 2.0]]))


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_auto_falls_back_to_cpu_without_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA available in this environment; auto would resolve to cuda")
    assert resolve_device("auto") == "cpu"


def test_resolve_device_passes_through_explicit_choice() -> None:
    assert resolve_device("cpu") == "cpu"


# ---------------------------------------------------------------------------
# EmbeddingMoodModel — CA-05 (paridade memória/disco), pickle sem torch
# ---------------------------------------------------------------------------


def _toy_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_clean": ["tok1 tok2", "tok3 tok4"],
            "context_clean": [["tok5 tok6"], []],
        }
    )


def test_embedding_mood_model_predict_shape_and_range(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    settings = EncoderSettings(source="tiny", revision="n/a", max_length=16, device="cpu", num_threads=None)
    model = EmbeddingMoodModel(
        settings, context_weight=0.5, coef=np.zeros(encoder.config.hidden_size * 2), intercept=0.0
    ).bind(encoder, tokenizer)

    scores = model.predict(_toy_dataframe())

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_embedding_mood_model_pickle_never_contains_torch(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    settings = EncoderSettings(source="tiny", revision="n/a", max_length=16, device="cpu", num_threads=None)
    model = EmbeddingMoodModel(
        settings, context_weight=0.5, coef=np.ones(encoder.config.hidden_size * 2), intercept=0.1
    ).bind(encoder, tokenizer)

    payload = pickle.dumps(model)

    assert b"torch" not in payload


def test_embedding_mood_model_disk_roundtrip_matches_in_memory_prediction(tmp_path: Path) -> None:
    """CA-05: para o mesmo history, a predição do wrapper carregado do disco é
    igual à do modelo em memória logo após o treino."""
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    settings = EncoderSettings(source="tiny", revision="n/a", max_length=16, device="cpu", num_threads=None)
    model = EmbeddingMoodModel(
        settings, context_weight=0.5, coef=np.linspace(-1, 1, encoder.config.hidden_size * 2), intercept=0.05
    ).bind(encoder, tokenizer)

    df = _toy_dataframe()
    in_memory_scores = model.predict(df)

    encoder_dir = tmp_path / "encoder"
    model.save_encoder(encoder_dir)
    reloaded: EmbeddingMoodModel = pickle.loads(pickle.dumps(model))
    reloaded.attach(encoder_dir)

    disk_scores = reloaded.predict(df)

    assert np.allclose(in_memory_scores, disk_scores, atol=1e-6)


def test_embedding_mood_model_predict_without_bind_raises() -> None:
    settings = EncoderSettings(source="tiny", revision="n/a", max_length=16, device="cpu", num_threads=None)
    model = EmbeddingMoodModel(settings, context_weight=0.5, coef=np.zeros(4), intercept=0.0)

    with pytest.raises(RuntimeError):
        model.predict(_toy_dataframe())
