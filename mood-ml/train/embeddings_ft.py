"""Loop de treino da abordagem C: sondagem linear + fine-tuning (see specs/11-embeddings-finetuning.md §3).

Fica fora de `transform/` porque carrega otimizador e *scheduler* — código só
de treino, nunca de inferência (P4). Nunca importa `train.train`: quando
`train/train.py` roda como `python -m train.train`, o módulo é carregado duas
vezes sob nomes diferentes (`__main__` e `train.train`); importar daqui
criaria uma segunda cópia de qualquer estado definido lá (mesmo motivo que
mantém `join_context_list` em `transform/features.py`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import nn

from evaluate.metrics import clip_score
from transform.embeddings import build_features, encode_context, encode_texts

_NOOP_EMIT: Callable[[str, dict[str, Any]], None] = lambda event, fields: None


@dataclass
class FineTuneSettings:
    """Espelha `train.embeddings.finetune` em configs/pipeline.yaml (spec 11 §4)."""

    epochs_max: int
    patience: int
    batch_size: int
    lr_encoder: float
    lr_head: float
    weight_decay: float
    warmup_ratio: float
    dropout: float
    freeze_word_embeddings: bool


def seed_everything(seed: int) -> torch.Generator:
    """FT-R12: todo componente estocástico recebe a seed do config. Devolve um
    `Generator` dedicado ao embaralhamento dos batches (não usamos `DataLoader`:
    o embaralhamento é manual, ver `_iter_batches`)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return torch.Generator().manual_seed(seed)


def _compute_features(
    encoder: Any, tokenizer: Any, df: pd.DataFrame, max_length: int, weight: float, device: str
) -> np.ndarray:
    texts = df["text_clean"].tolist()
    context_lists = [list(row) for row in df["context_clean"]]
    with torch.no_grad():
        text_embeddings = encode_texts(encoder, tokenizer, texts, max_length, device)
        context_embeddings = encode_context(encoder, tokenizer, context_lists, max_length, device)
        features = build_features(text_embeddings, context_embeddings, weight)
    return features.cpu().numpy()


def _iter_batches(n: int, batch_size: int, generator: torch.Generator) -> Iterator[np.ndarray]:
    indices = torch.randperm(n, generator=generator).numpy()
    for start in range(0, n, batch_size):
        yield indices[start : start + batch_size]


# ---------------------------------------------------------------------------
# Etapa 1 — sondagem linear (FT-R10)
# ---------------------------------------------------------------------------


def run_probe(
    encoder: Any,
    tokenizer: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_length: int,
    context_weight: float,
    device: str,
    probe_alpha: float,
    probe_solver: str,
) -> tuple[np.ndarray, float, float]:
    """Encoder congelado (sem gradiente, sem dropout); `Ridge` sobre
    `[e_t ; w·c]`. Devolve `(coef, intercept, probe_validation_mae)` — a cabeça
    do fine-tuning parte exatamente destes pesos (FT-R10)."""
    was_training = encoder.training
    encoder.eval()
    try:
        train_features = _compute_features(encoder, tokenizer, train_df, max_length, context_weight, device)
        validation_features = _compute_features(
            encoder, tokenizer, validation_df, max_length, context_weight, device
        )
    finally:
        encoder.train(was_training)

    ridge = Ridge(alpha=probe_alpha, solver=probe_solver)
    ridge.fit(train_features, train_df["label_score"].to_numpy())

    validation_preds = ridge.predict(validation_features)
    if not np.all(np.isfinite(validation_preds)):
        raise FloatingPointError("FT_R13_NON_FINITE: non-finite prediction during linear probing")
    clipped = np.array([clip_score(v) for v in validation_preds])
    probe_validation_mae = float(np.mean(np.abs(clipped - validation_df["label_score"].to_numpy())))

    return ridge.coef_.astype(np.float64), float(ridge.intercept_), probe_validation_mae


# ---------------------------------------------------------------------------
# Etapa 2 — fine-tuning (FT-R09, FT-R11 a FT-R13)
# ---------------------------------------------------------------------------


def finetune(
    encoder: Any,
    tokenizer: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_length: int,
    context_weight: float,
    device: str,
    settings: FineTuneSettings,
    seed: int,
    probe_coef: np.ndarray,
    probe_intercept: float,
    emit: Callable[[str, dict[str, Any]], None] = _NOOP_EMIT,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """AdamW com dois grupos de parâmetros (`lr_encoder`/`lr_head`), *warmup*
    linear, *early stopping* no MAE de `validation` (recortado), melhor
    *checkpoint* restaurado ao final. Devolve `(coef, intercept, info)`."""
    generator = seed_everything(seed)
    encoder.to(device)
    encoder.train()

    if settings.freeze_word_embeddings:
        encoder.get_input_embeddings().weight.requires_grad_(False)

    hidden_size = encoder.config.hidden_size
    head = nn.Sequential(nn.Dropout(settings.dropout), nn.Linear(hidden_size * 2, 1)).to(device)
    with torch.no_grad():
        head[1].weight.copy_(torch.as_tensor(probe_coef, dtype=head[1].weight.dtype).unsqueeze(0))
        head[1].bias.copy_(torch.as_tensor([probe_intercept], dtype=head[1].bias.dtype))

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": settings.lr_encoder},
            {"params": head.parameters(), "lr": settings.lr_head},
        ],
        weight_decay=settings.weight_decay,
    )

    n_train = len(train_df)
    steps_per_epoch = max(1, (n_train + settings.batch_size - 1) // settings.batch_size)
    total_steps = settings.epochs_max * steps_per_epoch
    warmup_steps = max(1, int(total_steps * settings.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / warmup_steps))

    train_labels = train_df["label_score"].to_numpy()
    validation_labels = validation_df["label_score"].to_numpy()

    def _evaluate_validation_mae() -> float:
        was_training = encoder.training
        encoder.eval()
        head.eval()
        try:
            with torch.no_grad():
                features = _compute_features(
                    encoder, tokenizer, validation_df, max_length, context_weight, device
                )
                preds = head(torch.as_tensor(features, dtype=torch.float32, device=device)).squeeze(-1).cpu().numpy()
        finally:
            encoder.train(was_training)
            head.train(was_training)
        if not np.all(np.isfinite(preds)):
            raise FloatingPointError("FT_R13_NON_FINITE: non-finite prediction on validation")
        clipped = np.array([clip_score(v) for v in preds])
        return float(np.mean(np.abs(clipped - validation_labels)))

    best_mae = _evaluate_validation_mae()  # época 0: antes de qualquer passo de gradiente (CA-09)
    best_state = {
        "encoder": {key: value.detach().clone() for key, value in encoder.state_dict().items()},
        "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
    }
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(settings.epochs_max):
        encoder.train()
        head.train()
        for batch_indices in _iter_batches(n_train, settings.batch_size, generator):
            batch_df = train_df.iloc[batch_indices]
            texts = batch_df["text_clean"].tolist()
            context_lists = [list(row) for row in batch_df["context_clean"]]

            text_embeddings = encode_texts(encoder, tokenizer, texts, max_length, device)
            context_embeddings = encode_context(encoder, tokenizer, context_lists, max_length, device)
            features = build_features(text_embeddings, context_embeddings, context_weight)
            preds = head(features).squeeze(-1)

            targets = torch.as_tensor(train_labels[batch_indices], dtype=torch.float32, device=device)
            loss = nn.functional.mse_loss(preds, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("FT_R13_NON_FINITE: non-finite training loss")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        epochs_run = epoch + 1
        validation_mae = _evaluate_validation_mae()
        emit("epoch_finished", {"epoch": epochs_run, "validation_mae": validation_mae})

        if validation_mae < best_mae - 1e-12:
            best_mae = validation_mae
            best_state = {
                "encoder": {key: value.detach().clone() for key, value in encoder.state_dict().items()},
                "head": {key: value.detach().clone() for key, value in head.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.patience:
                break

    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])

    coef = head[1].weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
    intercept = float(head[1].bias.detach().cpu().item())
    info = {"finetune_validation_mae": best_mae, "epochs_run": epochs_run}
    return coef, intercept, info


# ---------------------------------------------------------------------------
# Orquestração das duas etapas
# ---------------------------------------------------------------------------


def fit_embeddings_ft(
    encoder: Any,
    tokenizer: Any,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    max_length: int,
    context_weight: float,
    probe_alpha: float,
    probe_solver: str,
    finetune_settings: FineTuneSettings,
    seed: int,
    device: str,
    emit: Callable[[str, dict[str, Any]], None] = _NOOP_EMIT,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Roda `run_probe` e então `finetune`, inicializando a cabeça com os pesos
    da sondagem (FT-R10). Devolve `(coef, intercept, info)`; `info` alimenta
    `train_manifest.json` (FT-R14)."""
    probe_coef, probe_intercept, probe_validation_mae = run_probe(
        encoder, tokenizer, train_df, validation_df, max_length, context_weight, device, probe_alpha, probe_solver
    )
    emit("probe_finished", {"probe_validation_mae": probe_validation_mae})

    coef, intercept, info = finetune(
        encoder,
        tokenizer,
        train_df,
        validation_df,
        max_length,
        context_weight,
        device,
        finetune_settings,
        seed,
        probe_coef,
        probe_intercept,
        emit,
    )
    info = {
        **info,
        "probe_validation_mae": probe_validation_mae,
        "device": device,
        "torch_version": torch.__version__,
    }
    return coef, intercept, info
