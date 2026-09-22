import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from infer.predict import (
    FeatureSpecMismatchError,
    HistoryMessage,
    RegistryBrokenError,
    load_active_model,
    run_inference,
)
from tests.conftest import register_active_model


def _msg(message_id: str, role: str = "customer", text: str = "bom dia, tudo otimo") -> HistoryMessage:
    return HistoryMessage(message_id=message_id, role=role, text=text, sent_at=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# load_active_model — IF-R01 a IF-R04
# ---------------------------------------------------------------------------


def test_no_active_json_returns_degraded_state_without_raising(tmp_path: Path) -> None:
    state = load_active_model(tmp_path / "models")
    assert state.pipeline is None
    assert state.model_version is None
    assert state.manifest is None


def test_happy_path_loads_compatible_model(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-a", train_config)
    state = load_active_model(corpus_root / "models")
    assert state.pipeline is not None
    assert state.model_version == model_version
    assert state.manifest["algorithm"] == "tfidf-ridge"


def test_registry_broken_when_manifest_missing(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-b", train_config)
    (corpus_root / "models" / model_version / "manifest.json").unlink()

    with pytest.raises(RegistryBrokenError) as exc:
        load_active_model(corpus_root / "models")
    assert exc.value.code == "IF_R03_STARTUP_REGISTRY_BROKEN"


def test_registry_broken_when_model_joblib_missing(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-c", train_config)
    (corpus_root / "models" / model_version / "model.joblib").unlink()

    with pytest.raises(RegistryBrokenError):
        load_active_model(corpus_root / "models")


def test_registry_broken_when_manifest_json_corrupted(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-d", train_config)
    (corpus_root / "models" / model_version / "manifest.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(RegistryBrokenError):
        load_active_model(corpus_root / "models")


def test_registry_broken_when_model_joblib_corrupted(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-e", train_config)
    (corpus_root / "models" / model_version / "model.joblib").write_bytes(b"not a real joblib file")

    with pytest.raises(RegistryBrokenError):
        load_active_model(corpus_root / "models")


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("feature_spec_version", "fs-999"),
        ("history_window", 1),
        ("history_scope", "customer"),
        ("algorithm", "embeddings-ridge"),
    ],
)
def test_feature_spec_mismatch_blocks_startup(
    corpus_root: Path, train_config: dict, field: str, bad_value: object
) -> None:
    model_version = register_active_model(corpus_root, "ds-f", train_config)
    manifest_path = corpus_root / "models" / model_version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = bad_value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FeatureSpecMismatchError) as exc:
        load_active_model(corpus_root / "models")
    assert exc.value.field == field
    assert exc.value.code == "IF_R04_STARTUP_FEATURE_SPEC_MISMATCH"


# ---------------------------------------------------------------------------
# run_inference — determinismo, clip_score, extract_features sempre chamada
# ---------------------------------------------------------------------------


def test_run_inference_is_deterministic(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-g", train_config)
    state = load_active_model(corpus_root / "models")

    history = [_msg("m1")]
    score1 = run_inference(state.pipeline, history)
    score2 = run_inference(state.pipeline, history)
    assert score1 == score2


def test_run_inference_clips_out_of_range_predictions() -> None:
    class _FakePipeline:
        def predict(self, _df):
            return np.array([5.0])

    score = run_inference(_FakePipeline(), [_msg("m1")])
    assert score == 1.0

    class _FakePipelineNegative:
        def predict(self, _df):
            return np.array([-7.0])

    score_neg = run_inference(_FakePipelineNegative(), [_msg("m1")])
    assert score_neg == -1.0


def test_run_inference_always_calls_extract_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """IF-R21: extract_features roda para toda requisição válida, mesmo que
    o texto já pareça mascarado."""
    import infer.predict as predict_mod

    calls = []
    real_extract_features = predict_mod.extract_features

    def _spy(history):
        calls.append(history)
        return real_extract_features(history)

    monkeypatch.setattr(predict_mod, "extract_features", _spy)

    class _FakePipeline:
        def predict(self, _df):
            return np.array([0.0])

    history = [_msg("m1", text="ja mascarado <EMAIL>")]
    run_inference(_FakePipeline(), history)

    assert len(calls) == 1
    assert calls[0] == [{"role": "customer", "text": "ja mascarado <EMAIL>"}]
