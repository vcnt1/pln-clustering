import time
from pathlib import Path

import pytest

from evaluate.metrics import EvaluateGateError, clip_score, evaluate_candidate
from tests.conftest import build_synthetic_dataset, train_and_stage

# ---------------------------------------------------------------------------
# clip_score (CA-08)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (1.4, 1.0),
        (-1.2, -1.0),
        (0.3, 0.3),
        (-0.9, -0.9),
        (1.0, 1.0),
        (-1.0, -1.0),
    ],
)
def test_clip_score(raw: float, expected: float) -> None:
    assert clip_score(raw) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_gate_blocked_when_staging_incomplete(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-x")
    with pytest.raises(EvaluateGateError) as exc:
        evaluate_candidate("ds-x", train_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets")
    assert exc.value.code == "EV_R01_GATE_STAGING_NOT_OK"


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_mae_by_persona_covers_all_test_personas(corpus_root: Path, train_config: dict) -> None:
    build_synthetic_dataset(corpus_root, "ds-y", signal="strong")
    train_and_stage(corpus_root, "ds-y", train_config)
    result = evaluate_candidate("ds-y", train_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets")

    import pandas as pd

    test_df = pd.read_parquet(corpus_root / "datasets" / "ds-y" / "test.parquet")
    assert set(result.eval_json["mae_by_persona"].keys()) == set(test_df["persona"].unique())


def test_context_breakdown_handles_all_no_context_without_division_by_zero(
    corpus_root: Path, train_config: dict
) -> None:
    """CA-11: a dataset where every test example has context_clean == []
    must not raise, and with_context_rows/mae must be 0/None."""
    dataset_dir = build_synthetic_dataset(corpus_root, "ds-z", signal="strong")

    import pandas as pd

    test_path = dataset_dir / "test.parquet"
    test_df = pd.read_parquet(test_path)
    test_df["context_clean"] = [[] for _ in range(len(test_df))]
    test_df.to_parquet(test_path, index=False)

    train_and_stage(corpus_root, "ds-z", train_config)
    result = evaluate_candidate("ds-z", train_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets")

    breakdown = result.eval_json["context_breakdown"]
    assert breakdown["with_context_rows"] == 0
    assert breakdown["with_context_mae"] is None
    assert breakdown["no_context_rows"] == len(test_df)
    assert breakdown["no_context_mae"] is not None


# ---------------------------------------------------------------------------
# Latência — item a item, nunca em lote (EV-R06/CA-12)
# ---------------------------------------------------------------------------


def test_latency_is_measured_item_by_item_not_batched(
    corpus_root: Path, train_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    import evaluate.metrics as evaluate_mod

    build_synthetic_dataset(corpus_root, "ds-lat", signal="strong", n_test=6)
    train_and_stage(corpus_root, "ds-lat", train_config)

    real_load = evaluate_mod.joblib.load
    call_sizes: list[int] = []
    delay_s = 0.02

    def _instrumented_load(path):
        obj = real_load(path)
        if Path(path).name == "candidate.joblib":
            real_predict = obj.predict

            def _slow_predict(X):
                call_sizes.append(len(X))
                time.sleep(delay_s)
                return real_predict(X)

            obj.predict = _slow_predict
        return obj

    monkeypatch.setattr(evaluate_mod.joblib, "load", _instrumented_load)

    result = evaluate_candidate(
        "ds-lat", train_config, staging_dir=corpus_root / "staging", datasets_dir=corpus_root / "datasets"
    )

    # The metrics step calls predict() once, batched, over the whole test
    # set; the latency loop must call it once per row after that.
    single_row_calls = [size for size in call_sizes if size == 1]
    assert len(single_row_calls) == 6

    latency = result.eval_json["latency"]
    assert latency["sample_size"] == 6
    # Loose bound: each of the 6 calls slept ~20ms, so p50 should reflect a
    # single call's delay, not 6x it (which batching would produce instead).
    assert latency["p50_ms"] >= delay_s * 1000 * 0.5
    assert latency["p50_ms"] < delay_s * 1000 * 3
