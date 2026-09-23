"""[8] Online inference service: loads the active model and serves predictions (see specs/08-infer.md)."""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from evaluate.metrics import clip_score
from train.train import SUPPORTED_ALGORITHMS
from transform.features import FEATURE_SPEC_VERSION, MAX_HISTORY_SIZE, extract_features

IO_RETRY_BACKOFF_SECONDS = (1, 2, 4)
EXPECTED_HISTORY_WINDOW = MAX_HISTORY_SIZE
EXPECTED_HISTORY_SCOPE = "conversation"

logger = logging.getLogger("infer.predict")


# ---------------------------------------------------------------------------
# Contrato HTTP — data-model.md §2.2
# ---------------------------------------------------------------------------


class HistoryMessage(BaseModel):
    message_id: str
    role: str  # não Literal["customer"]: uma role diferente é regra de negócio
    #            (IF-R08, 400 invalid_history), não violação estrutural (400 invalid_request).
    text: str


class InferRequest(BaseModel):
    request_id: str
    customer_id: str
    conversation_id: str
    history: list[HistoryMessage] = Field(min_length=1, max_length=30)


class InferResponse(BaseModel):
    request_id: str
    customer_id: str
    conversation_id: str
    trigger_message_id: str
    score: float
    scale: str
    mood_label: str | None = None
    model_version: str
    computed_at: datetime


# ---------------------------------------------------------------------------
# Errors — subida do serviço
# ---------------------------------------------------------------------------


class RegistryBrokenError(Exception):
    """IF-R03: active.json aponta para um model_version sem manifest.json
    legível ou model.joblib válido."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.code = "IF_R03_STARTUP_REGISTRY_BROKEN"
        self.detail = detail


class FeatureSpecMismatchError(Exception):
    """IF-R04: manifest.feature_spec_version/history_window/history_scope/
    algorithm diverge do que este código implementa (P4)."""

    def __init__(self, field: str, expected: Any, found: Any) -> None:
        detail = f"{field} mismatch: expected={expected!r} found={found!r}"
        super().__init__(detail)
        self.code = "IF_R04_STARTUP_FEATURE_SPEC_MISMATCH"
        self.field = field
        self.expected = expected
        self.found = found
        self.detail = detail


class _InferIOError(Exception):
    """Sinal interno de _with_io_retry — nunca escapa de load_active_model,
    sempre remapeado para RegistryBrokenError (é sempre um estado de
    'registro quebrado', tenha a causa sido ausência de arquivo, corrupção
    de conteúdo ou uma falha transitória que não se resolveu nas 3 tentativas)."""


def _with_io_retry(operation: str, func: Any) -> Any:
    last_exc: OSError | None = None
    for attempt, backoff in enumerate((0, *IO_RETRY_BACKOFF_SECONDS), start=1):
        if backoff:
            time.sleep(backoff)
        try:
            return func()
        except OSError as exc:
            last_exc = exc
            _log(logging.WARNING, "io_retry", attempt=attempt, operation=operation, errno=exc.errno)
    raise _InferIOError(f"{operation} failed after retries: {last_exc}")


# ---------------------------------------------------------------------------
# load_active_model — IF-R01 a IF-R05, IF-R19
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    pipeline: Any | None
    model_version: str | None
    manifest: dict[str, Any] | None


def _check_compatibility(manifest: dict[str, Any]) -> None:
    if manifest.get("feature_spec_version") != FEATURE_SPEC_VERSION:
        raise FeatureSpecMismatchError(
            "feature_spec_version", FEATURE_SPEC_VERSION, manifest.get("feature_spec_version")
        )
    if manifest.get("history_window") != EXPECTED_HISTORY_WINDOW:
        raise FeatureSpecMismatchError("history_window", EXPECTED_HISTORY_WINDOW, manifest.get("history_window"))
    if manifest.get("history_scope") != EXPECTED_HISTORY_SCOPE:
        raise FeatureSpecMismatchError("history_scope", EXPECTED_HISTORY_SCOPE, manifest.get("history_scope"))
    if manifest.get("algorithm") not in SUPPORTED_ALGORITHMS:
        raise FeatureSpecMismatchError("algorithm", sorted(SUPPORTED_ALGORITHMS), manifest.get("algorithm"))


def load_active_model(models_dir: str | Path) -> ModelState:
    """IF-R01/R02: active.json ausente -> ModelState degradado (pipeline=None),
    sem exceção — estado legítimo de bootstrap. IF-R03: manifest.json/
    model.joblib ausentes ou ilegíveis -> RegistryBrokenError. IF-R04:
    manifesto incompatível com o código -> FeatureSpecMismatchError. IF-R19:
    toda leitura real (não a checagem de existência) passa por retry 3x,
    1s/2s/4s."""
    models_dir = Path(models_dir)
    active_path = models_dir / "active.json"
    if not active_path.exists():
        return ModelState(pipeline=None, model_version=None, manifest=None)

    try:
        active = _with_io_retry(
            "read_active_json", lambda: json.loads(active_path.read_text(encoding="utf-8"))
        )
    except (_InferIOError, json.JSONDecodeError) as exc:
        raise RegistryBrokenError(f"could not read {active_path}: {exc}") from exc

    model_version = active.get("model_version")
    model_dir = models_dir / str(model_version)
    manifest_path = model_dir / "manifest.json"
    model_path = model_dir / "model.joblib"

    # Checagem de existência simples (sem retentativa) — arquivo ausente é
    # um estado de registro quebrado, não uma falha transitória de I/O a
    # que valha a pena esperar 7s (mesma separação gate/leitura das Fases 1-3).
    if not manifest_path.exists() or not model_path.exists():
        raise RegistryBrokenError(f"incomplete model artifacts for model_version={model_version} at {model_dir}")

    try:
        manifest = _with_io_retry(
            "read_manifest_json", lambda: json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        pipeline = _with_io_retry("load_model_joblib", lambda: joblib.load(model_path))
    except _InferIOError as exc:
        raise RegistryBrokenError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise RegistryBrokenError(f"corrupted manifest.json at {manifest_path}: {exc}") from exc
    except Exception as exc:
        # Deliberate catch-all: joblib/pickle raise many exception types on a
        # corrupted artifact (EOFError, UnpicklingError, ValueError, ...).
        raise RegistryBrokenError(f"corrupted model.joblib at {model_path}: {exc}") from exc

    _check_compatibility(manifest)

    return ModelState(pipeline=pipeline, model_version=model_version, manifest=manifest)


# ---------------------------------------------------------------------------
# run_inference — passos 4-6 do §2.3, puro o suficiente para testar sem HTTP
# ---------------------------------------------------------------------------


def run_inference(pipeline: Any, history: list[HistoryMessage]) -> float:
    """extract_features roda sempre (IF-R21), mesmo que o texto já chegue
    mascarado — mask_pii é idempotente (spec 04). Nenhum I/O (IF-R18)."""
    items = [{"role": item.role, "text": item.text} for item in history]
    features = extract_features(items)
    row = pd.DataFrame([{
        "text_clean": features["text_clean"],
        "context_clean": features["context_clean"],
        "context_text": " ".join(features["context_clean"]),
    }])
    raw = pipeline.predict(row)[0]
    return clip_score(raw)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post("/infer", response_model=InferResponse)
async def infer_mood(payload: InferRequest, request: Request) -> InferResponse | JSONResponse:
    model_state: ModelState = request.app.state.model_state

    # Passo 2 (§2.3): modo degradado responde antes de qualquer outra checagem.
    if model_state.pipeline is None:
        _log(
            logging.WARNING,
            "request_rejected",
            request_id=payload.request_id,
            error_code="ml_unavailable",
            detail="no active model loaded",
        )
        return JSONResponse(status_code=503, content={"error_code": "ml_unavailable", "detail": "no active model loaded"})

    # Passo 3: regras de negócio (IF-R08).
    invalid_role = any(item.role != "customer" for item in payload.history)
    if invalid_role:
        detail = "history item has role != 'customer'"
        _log(
            logging.WARNING,
            "request_rejected",
            request_id=payload.request_id,
            error_code="invalid_history",
            detail=detail,
        )
        return JSONResponse(status_code=400, content={"error_code": "invalid_history", "detail": detail})

    _log(
        logging.INFO,
        "request_received",
        request_id=payload.request_id,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        history_size=len(payload.history),
    )

    # Passos 4-6: nenhum I/O, tudo em memória (IF-R18).
    t0 = time.perf_counter()
    try:
        score = run_inference(model_state.pipeline, payload.history)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all mapped to 500 internal_error, never a score
        _log(
            logging.ERROR,
            "request_failed",
            request_id=payload.request_id,
            error_code="internal_error",
            exc_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=500, content={"error_code": "internal_error", "detail": "internal error computing the score"}
        )
    latency_ms = (time.perf_counter() - t0) * 1000

    response = InferResponse(
        request_id=payload.request_id,
        customer_id=payload.customer_id,
        conversation_id=payload.conversation_id,
        trigger_message_id=payload.history[-1].message_id,
        score=score,
        scale="-1 to 1",
        mood_label=None,
        model_version=model_state.model_version,
        computed_at=datetime.now(timezone.utc),
    )
    _log(
        logging.INFO,
        "request_completed",
        request_id=payload.request_id,
        model_version=model_state.model_version,
        score=score,
        latency_ms=latency_ms,
    )
    return response


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    model_state: ModelState = request.app.state.model_state
    if model_state.pipeline is None:
        return {"status": "degraded", "model_version": None}
    return {"status": "ok", "model_version": model_state.model_version}


# ---------------------------------------------------------------------------
# Logging — mesmo padrão duplicado por módulo das Fases 1-3, configurado uma
# vez na subida (não em um main() de CLI: este módulo não tem argv próprio).
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": _now_iso(), "level": record.levelname, "event": record.getMessage()}
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = getattr(record, "fields", None)
        suffix = f" {extra}" if extra else ""
        return f"{record.levelname:<7} {record.getMessage()}{suffix}"


def _configure_logging(log_format: str, log_level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if log_format == "json" else _TextFormatter())
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False


def _log(level: int, event: str, **fields: Any) -> None:
    run_id = fields.pop("run_id", None)
    payload = {"run_id": run_id, **fields} if run_id is not None else fields
    logger.log(level, event, extra={"fields": payload})
