"""pipeline.py: `all` ponta a ponta com abordagem C — spec 11 Fase 12 (checklist final).

Roda só com requirements-embeddings.txt instalado (marker `embeddings`).
`load_base_encoder` é substituído pelo BERT minúsculo de teste (fixture
`mock_base_encoder`, conftest.py) — o encoder real nunca é baixado aqui.
"""

import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import pipeline
from tests.conftest import embeddings_ft_config, write_corpus

pytestmark = pytest.mark.embeddings

MEDIUM_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_medium.jsonl"

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


def _stage_corpus(directory: Path, fixture_path: Path, corpus_id: str) -> Path:
    rows = [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line]
    return write_corpus(directory, corpus_id, rows)


def _all_args(tmp_path: Path, corpus_path: Path, **overrides: str) -> list[str]:
    args = [
        "all",
        str(corpus_path),
        "--skip-composition",
        "--report-dir", str(tmp_path / "reports"),
        "--labels-dir", str(tmp_path / "labels"),
        "--datasets-dir", str(tmp_path / "datasets"),
        "--staging-dir", str(tmp_path / "staging"),
        "--models-dir", str(tmp_path / "models"),
    ]
    for flag, value in overrides.items():
        args.extend([f"--{flag.replace('_', '-')}", value])
    return args


def _write_embeddings_config(tmp_path: Path) -> Path:
    """Parte do configs/pipeline.yaml real (para herdar o bloco `split`) e só
    troca `train` para embeddings-ft — o `revision` real não importa aqui
    porque `load_base_encoder` está mockado (fixture `mock_base_encoder`)."""
    base_config = yaml.safe_load(Path("configs/pipeline.yaml").read_text(encoding="utf-8"))
    config = embeddings_ft_config(base_config, finetune=_FAST_FINETUNE)
    config["evaluate"] = {"quality_gate": {"max_mae_ratio": 10.0}}
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_all_runs_end_to_end_with_embeddings_ft(tmp_path: Path, mock_base_encoder: None) -> None:
    corpus_path = _stage_corpus(tmp_path, MEDIUM_FIXTURE, "syn-2026-01-02-a")
    config_path = _write_embeddings_config(tmp_path)

    code = pipeline.main(_all_args(tmp_path, corpus_path, config=str(config_path)))
    assert code == 0

    model_dirs = list((tmp_path / "models").glob("mood-*"))
    assert len(model_dirs) == 1
    assert (model_dirs[0] / "model.joblib").exists()
    assert (model_dirs[0] / "encoder" / "config.json").exists()
    assert (model_dirs[0] / "encoder" / "tokenizer_config.json").exists()

    manifest = json.loads((model_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithm"] == "embeddings-ft"
    assert set(manifest["encoder"]) == {"source", "revision", "max_length"}

    # active.json nunca é criado por `all` (D4/PL-R08) — promote é sempre manual,
    # igual para A e C.
    assert not (tmp_path / "models" / "active.json").exists()
