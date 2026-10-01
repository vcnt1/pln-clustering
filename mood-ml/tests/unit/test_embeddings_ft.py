"""Loop de treino da abordagem C (spec 11 §3): sondagem linear e fine-tuning.

Roda só com requirements-embeddings.txt instalado (pytest.ini: marker
`embeddings`); pulado via importorskip quando torch/transformers não estão
presentes (CA-06).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.conftest import build_tiny_encoder
from train.embeddings_ft import (
    FineTuneSettings,
    _compute_features,
    finetune,
    run_probe,
)
from transform.embeddings import (
    build_features,
    encode_context,
    encode_texts,
)

pytestmark = pytest.mark.embeddings

MAX_LENGTH = 16
CONTEXT_WEIGHT = 0.5


def _toy_dataset(n_train: int = 12, n_validation: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)

    def _make(n_rows: int, offset: int) -> pd.DataFrame:
        texts = [f"tok{(i + offset) % 90} tok{(i + offset + 1) % 90}" for i in range(n_rows)]
        contexts = [[f"tok{(i + offset + 2) % 90}"] if i % 2 == 0 else [] for i in range(n_rows)]
        labels = rng.uniform(-1, 1, size=n_rows)
        return pd.DataFrame({"text_clean": texts, "context_clean": contexts, "label_score": labels})

    return _make(n_train, 0), _make(n_validation, 1000)


def _default_settings(**overrides: object) -> FineTuneSettings:
    base = {
        "epochs_max": 2,
        "patience": 2,
        "batch_size": 4,
        "lr_encoder": 1e-3,
        "lr_head": 1e-2,
        "weight_decay": 0.0,
        "warmup_ratio": 0.2,
        "dropout": 0.1,
        "freeze_word_embeddings": True,
    }
    base.update(overrides)
    return FineTuneSettings(**base)


# ---------------------------------------------------------------------------
# run_probe
# ---------------------------------------------------------------------------


def test_run_probe_returns_finite_coef_and_mae(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    train_df, validation_df = _toy_dataset()

    coef, intercept, mae = run_probe(
        encoder, tokenizer, train_df, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu", 1.0, "lsqr"
    )

    assert coef.shape == (encoder.config.hidden_size * 2,)
    assert np.isfinite(intercept)
    assert np.isfinite(mae)


# ---------------------------------------------------------------------------
# CA-09 — época 0 do fine-tuning == sondagem
# ---------------------------------------------------------------------------


def test_finetune_epoch_zero_matches_probe_validation_mae(tmp_path: Path) -> None:
    """CA-09: a cabeça inicializada pela sondagem (etapa 1) dá, antes de
    qualquer passo de gradiente, o mesmo MAE de validation que a sondagem."""
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    train_df, validation_df = _toy_dataset()
    coef, intercept, probe_mae = run_probe(
        encoder, tokenizer, train_df, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu", 1.0, "lsqr"
    )

    settings = _default_settings(epochs_max=0)
    _, _, info = finetune(
        encoder,
        tokenizer,
        train_df,
        validation_df,
        MAX_LENGTH,
        CONTEXT_WEIGHT,
        "cpu",
        settings,
        seed=42,
        probe_coef=coef,
        probe_intercept=intercept,
    )

    assert info["epochs_run"] == 0
    assert info["finetune_validation_mae"] == pytest.approx(probe_mae, abs=1e-5)


# ---------------------------------------------------------------------------
# FT-R09 — matriz de palavras congelada
# ---------------------------------------------------------------------------


def test_finetune_freezes_word_embeddings_when_requested(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    train_df, validation_df = _toy_dataset()
    coef, intercept, _ = run_probe(
        encoder, tokenizer, train_df, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu", 1.0, "lsqr"
    )
    original_embeddings = encoder.get_input_embeddings().weight.detach().clone()

    settings = _default_settings(freeze_word_embeddings=True, lr_encoder=1.0, lr_head=1.0, dropout=0.0)
    finetune(
        encoder,
        tokenizer,
        train_df,
        validation_df,
        MAX_LENGTH,
        CONTEXT_WEIGHT,
        "cpu",
        settings,
        seed=42,
        probe_coef=coef,
        probe_intercept=intercept,
    )

    assert torch.equal(encoder.get_input_embeddings().weight, original_embeddings)
    assert encoder.get_input_embeddings().weight.requires_grad is False


def test_word_embeddings_receive_gradient_when_not_frozen(tmp_path: Path) -> None:
    """Complementa o teste acima testando o mecanismo diretamente (forward +
    backward manual), em vez de rodar `finetune()` ponta a ponta: com um
    `lr` alto o bastante para mudar os pesos de forma visível, o
    *early stopping* frequentemente não melhora sobre a época 0 e restaura o
    checkpoint inicial (FT-R11) — o que mascararia a mutação real durante o
    treino. Aqui a checagem é sobre o gradiente em si, não sobre o estado
    final pós-restauração."""
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    train_df, _ = _toy_dataset()
    texts = train_df["text_clean"].tolist()
    context_lists = [list(row) for row in train_df["context_clean"]]

    encoder.train()
    text_embeddings = encode_texts(encoder, tokenizer, texts, MAX_LENGTH, "cpu")
    context_embeddings = encode_context(encoder, tokenizer, context_lists, MAX_LENGTH, "cpu")
    features = build_features(text_embeddings, context_embeddings, CONTEXT_WEIGHT)
    features.sum().backward()

    grad = encoder.get_input_embeddings().weight.grad
    assert grad is not None
    assert torch.any(grad != 0)


# ---------------------------------------------------------------------------
# FT-R13 — perda não finita aborta sem persistir
# ---------------------------------------------------------------------------


def test_finetune_raises_floatingpointerror_on_non_finite_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    train_df, validation_df = _toy_dataset()
    coef, intercept, _ = run_probe(
        encoder, tokenizer, train_df, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu", 1.0, "lsqr"
    )
    monkeypatch.setattr(torch, "isfinite", lambda _x: torch.tensor(False))

    settings = _default_settings(epochs_max=1)
    with pytest.raises(FloatingPointError):
        finetune(
            encoder,
            tokenizer,
            train_df,
            validation_df,
            MAX_LENGTH,
            CONTEXT_WEIGHT,
            "cpu",
            settings,
            seed=42,
            probe_coef=coef,
            probe_intercept=intercept,
        )


# ---------------------------------------------------------------------------
# CA-03 — determinismo em CPU
# ---------------------------------------------------------------------------


def test_finetune_is_deterministic_on_cpu_with_same_seed(tmp_path: Path) -> None:
    encoder, tokenizer = build_tiny_encoder(tmp_path)
    initial_state = {key: value.clone() for key, value in encoder.state_dict().items()}
    train_df, validation_df = _toy_dataset()
    settings = _default_settings(freeze_word_embeddings=False)

    def _run() -> np.ndarray:
        encoder.load_state_dict(initial_state)
        coef, intercept, _ = run_probe(
            encoder, tokenizer, train_df, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu", 1.0, "lsqr"
        )
        final_coef, final_intercept, _ = finetune(
            encoder,
            tokenizer,
            train_df,
            validation_df,
            MAX_LENGTH,
            CONTEXT_WEIGHT,
            "cpu",
            settings,
            seed=123,
            probe_coef=coef,
            probe_intercept=intercept,
        )
        features = _compute_features(encoder, tokenizer, validation_df, MAX_LENGTH, CONTEXT_WEIGHT, "cpu")
        return features @ final_coef + final_intercept

    scores_1 = _run()
    scores_2 = _run()

    assert np.allclose(scores_1, scores_2, atol=1e-6)
