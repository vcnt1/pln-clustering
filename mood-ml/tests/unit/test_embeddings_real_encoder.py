"""Teste manual com o encoder real (spec 11 §9, "Estratégia de teste").

Baixa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` do
Hugging Face Hub — nunca roda em CI (marker `slow`, excluído por
`pytest -q -m "embeddings and not slow"`). Rodar manualmente com:

    pytest -q -m slow tests/unit/test_embeddings_real_encoder.py
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transform.embeddings import (
    EncoderSettings,
    build_features,
    encode_context,
    encode_texts,
    load_base_encoder,
)

pytestmark = [pytest.mark.embeddings, pytest.mark.slow]

# Mesma revision pinada em configs/pipeline.yaml (Q3, spec 11 §10).
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SOURCE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def test_real_encoder_produces_384_dim_embeddings() -> None:
    settings = EncoderSettings(source=SOURCE, revision=REVISION, max_length=128, device="cpu", num_threads=None)
    encoder, tokenizer = load_base_encoder(settings)

    assert encoder.config.hidden_size == 384

    text_embeddings = encode_texts(encoder, tokenizer, ["Qual o horário do check-out?"], 128, "cpu")
    assert text_embeddings.shape == (1, 384)

    context_embeddings = encode_context(encoder, tokenizer, [["Oi, tudo bem?", "Preciso de ajuda"]], 128, "cpu")
    assert context_embeddings.shape == (1, 384)

    features = build_features(text_embeddings, context_embeddings, weight=0.5)
    assert features.shape == (1, 768)
