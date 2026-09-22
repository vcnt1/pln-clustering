import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import create_app
from tests.conftest import register_active_model


def _history_item(i: int, role: str = "customer") -> dict:
    return {
        "message_id": f"msg-{i:03d}",
        "role": role,
        "text": f"mensagem {i} bom dia tudo otimo",
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }


def _payload(n_history: int = 1, trigger_message_id: str | None = None) -> dict:
    history = [_history_item(i) for i in range(n_history)]
    return {
        "request_id": "req-001",
        "customer_id": "cust-001",
        "conversation_id": "conv-001",
        "trigger_message_id": trigger_message_id or history[-1]["message_id"],
        "history": history,
    }


# ---------------------------------------------------------------------------
# CA-01 / CA-02 — subida ok e modo degradado
# ---------------------------------------------------------------------------


def test_ca01_valid_request_returns_score(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-a", train_config)
    app = create_app(models_dir=corpus_root / "models")

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=_payload())

    assert resp.status_code == 200
    body = resp.json()
    assert -1.0 <= body["score"] <= 1.0
    assert body["mood_label"] is None
    assert body["model_version"] == model_version
    assert body["scale"] == "-1 to 1"
    assert body["request_id"] == "req-001"


def test_ca02_degraded_mode_serves_but_infer_is_unavailable(tmp_path: Path) -> None:
    app = create_app(models_dir=tmp_path / "models")

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=_payload())
        health = client.get("/internal/v1/healthz")

    assert resp.status_code == 503
    assert resp.json()["error_code"] == "ml_unavailable"
    assert health.json() == {"status": "degraded", "model_version": None}


# ---------------------------------------------------------------------------
# CA-03 / CA-04 — recusa de subida
# ---------------------------------------------------------------------------


def test_ca03_registry_broken_refuses_to_start(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-b", train_config)
    (corpus_root / "models" / model_version / "manifest.json").unlink()

    app = create_app(models_dir=corpus_root / "models")
    # Lifespan startup failure — exact exception type is Starlette's, not ours.
    with pytest.raises(Exception), TestClient(app):  # noqa: B017
        pass


def test_ca04_feature_spec_mismatch_refuses_to_start(corpus_root: Path, train_config: dict) -> None:
    model_version = register_active_model(corpus_root, "ds-c", train_config)
    manifest_path = corpus_root / "models" / model_version / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["history_window"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    app = create_app(models_dir=corpus_root / "models")
    with pytest.raises(Exception), TestClient(app):  # noqa: B017
        pass


# ---------------------------------------------------------------------------
# CA-05 / CA-06 — validação de negócio
# ---------------------------------------------------------------------------


def test_ca05_history_item_with_wrong_role_returns_400(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-d", train_config)
    app = create_app(models_dir=corpus_root / "models")
    payload = _payload(n_history=2)
    payload["history"][0]["role"] = "agent"

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=payload)

    assert resp.status_code == 400
    assert resp.json()["error_code"] == "invalid_history"


def test_ca06_trigger_message_id_mismatch_returns_400(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-e", train_config)
    app = create_app(models_dir=corpus_root / "models")
    payload = _payload(trigger_message_id="not-the-last-message-id")

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=payload)

    assert resp.status_code == 400
    assert resp.json()["error_code"] == "invalid_history"


# ---------------------------------------------------------------------------
# CA-07 — validação estrutural (D5)
# ---------------------------------------------------------------------------


def test_ca07_missing_history_field_returns_custom_error_shape(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-f", train_config)
    app = create_app(models_dir=corpus_root / "models")
    payload = _payload()
    del payload["history"]

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=payload)

    assert resp.status_code == 400
    body = resp.json()
    assert set(body.keys()) == {"error_code", "detail"}
    assert body["error_code"] == "invalid_request"


# ---------------------------------------------------------------------------
# CA-08 / CA-09 — determinismo e ausência de score em erro
# ---------------------------------------------------------------------------


def test_ca08_repeated_calls_are_deterministic(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-g", train_config)
    app = create_app(models_dir=corpus_root / "models")
    payload = _payload()

    with TestClient(app) as client:
        resp1 = client.post("/internal/v1/infer", json=payload).json()
        resp2 = client.post("/internal/v1/infer", json=payload).json()

    resp1.pop("computed_at")
    resp2.pop("computed_at")
    assert resp1 == resp2


def test_ca09_internal_error_never_carries_a_score(
    corpus_root: Path, train_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    register_active_model(corpus_root, "ds-h", train_config)
    app = create_app(models_dir=corpus_root / "models")

    import infer.predict as predict_mod

    def _boom(pipeline, history):
        raise RuntimeError("boom")

    monkeypatch.setattr(predict_mod, "run_inference", _boom)

    with TestClient(app) as client:
        resp = client.post("/internal/v1/infer", json=_payload())

    assert resp.status_code == 500
    body = resp.json()
    assert body["error_code"] == "internal_error"
    assert "score" not in body


# ---------------------------------------------------------------------------
# CA-10 — latência local (janela cheia)
# ---------------------------------------------------------------------------


def test_ca10_latency_p95_is_plausible_with_a_full_window(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-i", train_config)
    app = create_app(models_dir=corpus_root / "models")
    payload = _payload(n_history=30)

    latencies_ms: list[float] = []
    with TestClient(app) as client:
        for _ in range(50):
            t0 = time.perf_counter()
            resp = client.post("/internal/v1/infer", json=payload)
            latencies_ms.append((time.perf_counter() - t0) * 1000)
            assert resp.status_code == 200

    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]
    # Meta da spec (D3) é p95 < 200ms medido pelo mood-api, sem a rede; aqui,
    # in-process via TestClient, é uma verificação de plausibilidade, não o
    # SLA de produção — limiar generoso para não ficar frágil em CI.
    assert p95 < 200


# ---------------------------------------------------------------------------
# CA-11 — nenhum texto de history vaza em log
# ---------------------------------------------------------------------------


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record.getMessage())
        extra = getattr(record, "fields", None)
        if extra:
            self.records.append(json.dumps(extra, default=str, ensure_ascii=False))


def test_ca11_pii_in_history_text_never_appears_in_logs(corpus_root: Path, train_config: dict) -> None:
    register_active_model(corpus_root, "ds-j", train_config)
    app = create_app(models_dir=corpus_root / "models")

    import infer.predict as predict_mod

    pii_text = "meu email eh joao@example.com e meu cpf 123.456.789-01"
    payload = _payload()
    payload["history"][-1]["text"] = pii_text

    handler = _ListHandler()
    with TestClient(app) as client:
        # Anexado depois da subida: _configure_logging (rodada no lifespan)
        # limpa os handlers do logger antes disso.
        predict_mod.logger.addHandler(handler)
        try:
            client.post("/internal/v1/infer", json=payload)
        finally:
            predict_mod.logger.removeHandler(handler)

    combined = " ".join(handler.records)
    assert pii_text not in combined
    assert "joao@example.com" not in combined


# ---------------------------------------------------------------------------
# CA-12 — P1
# ---------------------------------------------------------------------------


def test_ca12_no_duckdb_reference_in_infer() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "infer" / "predict.py"
    assert "duckdb" not in module_path.read_text(encoding="utf-8").lower()
