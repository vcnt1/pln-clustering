"""train.train: ramo C (embeddings-ft) ponta a ponta — spec 11 Fase 7.

Roda só com requirements-embeddings.txt instalado (marker `embeddings`).
`load_base_encoder` é substituído pelo BERT minúsculo de teste: a suíte nunca
depende do Hugging Face Hub estar acessível.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tests.conftest import (
    build_synthetic_dataset,
    embeddings_ft_config,
    train_and_stage,
)
from train.train import build_candidate

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
    """`epochs_max` pequeno para a suíte rodar rápido; o encoder é o BERT
    minúsculo de teste (fixture `mock_base_encoder`, conftest.py), não o real."""
    return embeddings_ft_config(train_config, finetune=_FAST_FINETUNE)


def test_embeddings_ft_build_candidate_succeeds_end_to_end(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    """CA-01: candidate.joblib, encoder/ e train_manifest.json com os campos
    de FT-R14."""
    build_synthetic_dataset(corpus_root, "ds-ft-e2e")
    config = _fast_config(train_config)

    staging_dir = train_and_stage(corpus_root, "ds-ft-e2e", config)

    assert (staging_dir / "baseline.joblib").exists()
    assert (staging_dir / "candidate.joblib").exists()
    assert (staging_dir / "encoder" / "config.json").exists()
    assert (staging_dir / "encoder" / "tokenizer_config.json").exists()

    manifest = json.loads((staging_dir / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "embeddings-ft"
    for field in ("device", "torch_version", "probe_validation_mae", "finetune_validation_mae", "epochs_run"):
        assert field in manifest
    # FT-R22: nenhum texto de cliente vaza para o manifesto.
    assert "text_clean" not in manifest
    assert "context_clean" not in manifest


def test_embeddings_ft_rerun_is_idempotent(corpus_root: Path, train_config: dict, mock_base_encoder: None) -> None:
    """CA-02: reexecução com o mesmo config é um no-op idempotente."""
    build_synthetic_dataset(corpus_root, "ds-ft-idem")
    config = _fast_config(train_config)

    staging_dir_1 = train_and_stage(corpus_root, "ds-ft-idem", config)
    result_2 = build_candidate(
        "ds-ft-idem", config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging"
    )

    assert result_2.reused is True
    assert result_2.staging_dir == staging_dir_1
    # O candidato recarregado do disco precisa estar pronto para prever
    # (attach() do encoder local), não só desserializado.
    assert result_2.candidate.predict is not None


def test_embeddings_ft_revision_change_creates_new_staging(
    corpus_root: Path, train_config: dict, mock_base_encoder: None
) -> None:
    """FT-R15: mudar `revision` gera staging novo; o anterior fica intacto."""
    build_synthetic_dataset(corpus_root, "ds-ft-revbump")
    config_a = _fast_config(train_config)
    config_b = _fast_config(train_config)
    config_b["train"]["embeddings"]["encoder"]["revision"] = "another-sha"

    staging_a = train_and_stage(corpus_root, "ds-ft-revbump", config_a)
    staging_b = train_and_stage(corpus_root, "ds-ft-revbump", config_b)

    assert staging_a != staging_b
    assert (staging_a / "candidate.joblib").exists()
    assert (staging_b / "candidate.joblib").exists()


def test_embeddings_ft_gate_blocks_before_any_parquet_read(
    corpus_root: Path, train_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CA-11, reforçado com dependências presentes (torch instalado neste
    teste): o gate F1b ainda aborta antes de ler qualquer parquet quando a
    config em si é inválida, mesmo com torch/transformers disponíveis."""
    import pandas as pd

    from train.train import TrainConfigError

    build_synthetic_dataset(corpus_root, "ds-ft-noread-2")
    config = _fast_config(train_config)
    del config["train"]["embeddings"]["probe"]["alpha"]

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parquet should not be read before the F1b gate")

    monkeypatch.setattr(pd, "read_parquet", _boom)

    with pytest.raises(TrainConfigError) as exc:
        build_candidate(
            "ds-ft-noread-2", config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging"
        )
    assert exc.value.code == "FT_R01_CONFIG_INVALID"


@pytest.mark.parametrize(
    ("block", "key", "bad_value"),
    [
        ("encoder", "device", "gpu"),  # fora de {"auto","cpu","cuda"}
        ("encoder", "max_length", 0),  # precisa ser int positivo
        ("encoder", "num_threads", -1),  # precisa ser int positivo ou None
        ("context", "weight", -0.5),  # precisa ser >= 0
        ("probe", "alpha", 0),  # precisa ser > 0
        ("probe", "solver", "not-a-solver"),  # fora do allow-list do Ridge
        ("finetune", "epochs_max", -1),  # precisa ser int positivo
        ("finetune", "patience", -1),  # precisa ser int >= 0
        ("finetune", "dropout", 5.0),  # fora de [0.0, 1.0]
        ("finetune", "warmup_ratio", -0.1),  # fora de [0.0, 1.0]
        ("finetune", "freeze_word_embeddings", "true"),  # precisa ser bool, não string
    ],
)
def test_embeddings_ft_config_rejects_malformed_values(
    corpus_root: Path,
    train_config: dict,
    monkeypatch: pytest.MonkeyPatch,
    block: str,
    key: str,
    bad_value: object,
) -> None:
    """FT-R01: a spec cobre chave 'ausente ou malformada' — valor de tipo/faixa
    inválida deve abortar com FT_R01_CONFIG_INVALID, exit 2, antes de ler
    qualquer parquet (mesmo gate F1b do teste acima)."""
    import pandas as pd

    from train.train import TrainConfigError

    build_synthetic_dataset(corpus_root, "ds-ft-malformed")
    config = _fast_config(train_config)
    config["train"]["embeddings"][block][key] = bad_value

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("parquet should not be read before the F1b gate")

    monkeypatch.setattr(pd, "read_parquet", _boom)

    with pytest.raises(TrainConfigError) as exc:
        build_candidate(
            "ds-ft-malformed", config, datasets_dir=corpus_root / "datasets", staging_dir=corpus_root / "staging"
        )
    assert exc.value.code == "FT_R01_CONFIG_INVALID"
