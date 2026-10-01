"""T3 (abordagem C): encoder de sentenças com fine-tuning (see specs/11-embeddings-finetuning.md).

Código compartilhado por treino (`train/embeddings_ft.py`) e inferência
(`infer/predict.py`) — P4 exige que tokenização, truncamento e pooling sejam a
mesma função nos dois caminhos (FT-R07). Só `train/embeddings_ft.py` importa
otimizador/scheduler; este módulo nunca treina nada.

Módulo de biblioteca, nunca ponto de entrada `-m` (mesmo motivo de
`join_context_list` em `transform/features.py`): `EmbeddingMoodModel` é
pickled por `joblib` e precisa resolver sempre para o mesmo módulo estável.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Config e carga do encoder
# ---------------------------------------------------------------------------


@dataclass
class EncoderSettings:
    """Só tipos primitivos — vai para dentro do pickle de `EmbeddingMoodModel`
    (spec 11 §5). Espelha `train.embeddings.encoder` em configs/pipeline.yaml."""

    source: str
    revision: str
    max_length: int
    device: str
    num_threads: int | None


def resolve_device(device_setting: str) -> str:
    if device_setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_setting


def load_base_encoder(settings: EncoderSettings) -> tuple[Any, Any]:
    """Baixa do Hugging Face Hub, pinado em `revision` (FT-R15). Só usado em
    treino — inferência nunca chama isto (FT-R17)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.source, revision=settings.revision)
    encoder = AutoModel.from_pretrained(settings.source, revision=settings.revision)
    return encoder, tokenizer


def load_local_encoder(encoder_dir: Path) -> tuple[Any, Any]:
    """FT-R17: carrega só do disco, `local_files_only=True`, sem acessar a rede."""
    from transformers import AutoModel, AutoTokenizer

    path = str(encoder_dir)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    encoder = AutoModel.from_pretrained(path, local_files_only=True)
    return encoder, tokenizer


# ---------------------------------------------------------------------------
# Pooling e codificação (FT-R05 a FT-R08)
# ---------------------------------------------------------------------------


def mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Média dos embeddings de token, ignorando posições de padding."""
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(encoder: Any, tokenizer: Any, texts: list[str], max_length: int, device: str) -> torch.Tensor:
    """FT-R07: tokenização, truncamento e pooling — a mesma função no treino
    (mensagem disparadora, com gradiente) e na inferência. Quem chama decide
    se roda dentro de `torch.no_grad()`; esta função não impõe isso, porque o
    fine-tuning precisa derivar por aqui para a mensagem disparadora."""
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    outputs = encoder(**encoded)
    return mean_pool(outputs.last_hidden_state, encoded["attention_mask"])


def encode_context(
    encoder: Any, tokenizer: Any, context_lists: list[list[str]], max_length: int, device: str
) -> torch.Tensor:
    """FT-R06/FT-R08: média dos embeddings das mensagens anteriores; vetor nulo
    quando a lista está vazia. Sempre sem gradiente, por construção — o
    contexto nunca participa do backward, só a mensagem disparadora."""
    hidden_size = encoder.config.hidden_size
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            flat_messages = [message for messages in context_lists for message in messages]
            if flat_messages:
                flat_embeddings = encode_texts(encoder, tokenizer, flat_messages, max_length, device)
            else:
                flat_embeddings = torch.empty((0, hidden_size), device=device)

            outputs = torch.zeros((len(context_lists), hidden_size), device=device)
            offset = 0
            for row, messages in enumerate(context_lists):
                count = len(messages)
                if count > 0:
                    outputs[row] = flat_embeddings[offset : offset + count].mean(dim=0)
                offset += count
            return outputs
    finally:
        encoder.train(was_training)


def build_features(text_embeddings: torch.Tensor, context_embeddings: torch.Tensor, weight: float) -> torch.Tensor:
    """[e_t ; w·c] — concatena a mensagem disparadora com o contexto ponderado."""
    return torch.cat([text_embeddings, weight * context_embeddings], dim=1)


# ---------------------------------------------------------------------------
# EmbeddingMoodModel — wrapper gravado em candidate.joblib / model.joblib
# ---------------------------------------------------------------------------


class EmbeddingMoodModel:
    """Wrapper leve persistido via joblib (spec 11 §5): hiperparâmetros, peso
    do contexto e os pesos numpy da cabeça linear. Nunca serializa tensores do
    torch — o encoder vive em `encoder/` ao lado (safetensors + tokenizer) e é
    vinculado de volta via `attach()` depois de carregar o pickle.

    Expõe `predict(df) -> np.ndarray` com as mesmas colunas de A (`text_clean`,
    `context_clean`), para `run_inference` funcionar sem mudança.
    """

    def __init__(self, encoder_settings: EncoderSettings, context_weight: float, coef: np.ndarray, intercept: float) -> None:
        self.encoder_settings = encoder_settings
        self.context_weight = context_weight
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self._encoder: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def bind(self, encoder: Any, tokenizer: Any) -> EmbeddingMoodModel:
        """Vincula um encoder/tokenizer já carregados em memória (treino, logo
        após o fine-tuning) — nunca pickled, só usado em processo."""
        if self.encoder_settings.num_threads is not None:
            torch.set_num_threads(self.encoder_settings.num_threads)
        self._device = resolve_device(self.encoder_settings.device)
        encoder.to(self._device)
        encoder.eval()
        self._encoder = encoder
        self._tokenizer = tokenizer
        return self

    def attach(self, encoder_dir: Path) -> EmbeddingMoodModel:
        """FT-R17: carrega o encoder local de `encoder_dir` e vincula."""
        encoder, tokenizer = load_local_encoder(encoder_dir)
        return self.bind(encoder, tokenizer)

    def save_encoder(self, dest_dir: Path) -> None:
        if self._encoder is None or self._tokenizer is None:
            raise RuntimeError("encoder not bound; call bind()/attach() first")
        dest_dir.mkdir(parents=True, exist_ok=True)
        self._encoder.save_pretrained(dest_dir, safe_serialization=True)
        self._tokenizer.save_pretrained(dest_dir)

    def predict(self, df: Any) -> np.ndarray:
        """Saída bruta (sem clip), no mesmo contrato do `Pipeline.predict()` de A
        (sklearn também não clipa): quem decide clipar é o chamador —
        `run_inference`, os diagnósticos de `train.train` e `evaluate.metrics`
        já fazem isso para A e continuam fazendo para C, sem mudança (spec 11
        §5, "mesmas colunas de A")."""
        if self._encoder is None or self._tokenizer is None:
            raise RuntimeError("encoder not bound; call bind()/attach() first")

        texts = df["text_clean"].tolist()
        context_lists = [list(row) for row in df["context_clean"]]

        with torch.no_grad():
            text_embeddings = encode_texts(
                self._encoder, self._tokenizer, texts, self.encoder_settings.max_length, self._device
            )
            context_embeddings = encode_context(
                self._encoder, self._tokenizer, context_lists, self.encoder_settings.max_length, self._device
            )
            features = build_features(text_embeddings, context_embeddings, self.context_weight)
            raw_scores = features.cpu().numpy() @ self.coef + self.intercept

        return raw_scores

    def __getstate__(self) -> dict[str, Any]:
        return {
            "encoder_settings": self.encoder_settings,
            "context_weight": self.context_weight,
            "coef": self.coef,
            "intercept": self.intercept,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.encoder_settings = state["encoder_settings"]
        self.context_weight = state["context_weight"]
        self.coef = state["coef"]
        self.intercept = state["intercept"]
        self._encoder = None
        self._tokenizer = None
        self._device = None
