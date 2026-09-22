"""ASGI entry point for the mood-ml online inference service (see specs/08-infer.md)."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from infer.predict import (
    FeatureSpecMismatchError,
    RegistryBrokenError,
    _configure_logging,
    _log,
    load_active_model,
)
from infer.predict import router as infer_router


def create_app(models_dir: str | Path = Path("models")) -> FastAPI:
    """Fábrica, não um `app` fixo de módulo: os testes de contrato de CA-02 a
    CA-04 precisam apontar para árvores de `models/` diferentes (degradado,
    registro quebrado, manifesto incompatível), e um único objeto de módulo
    com caminho fixo não permitiria isso."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _configure_logging(os.environ.get("LOG_FORMAT", "json"), os.environ.get("LOG_LEVEL", "INFO"))
        _log(logging.INFO, "startup_started")
        try:
            state = load_active_model(models_dir)
        except RegistryBrokenError as exc:
            _log(logging.ERROR, "startup_refused", reason=exc.code, detail=exc.detail)
            raise
        except FeatureSpecMismatchError as exc:
            _log(logging.ERROR, "startup_refused", reason=exc.code, expected=exc.expected, found=exc.found)
            raise

        if state.pipeline is None:
            _log(logging.WARNING, "startup_degraded", reason="active_json_missing")
        else:
            _log(
                logging.INFO,
                "model_loaded",
                model_version=state.model_version,
                algorithm=state.manifest.get("algorithm"),
                feature_spec_version=state.manifest.get("feature_spec_version"),
                history_window=state.manifest.get("history_window"),
                history_scope=state.manifest.get("history_scope"),
            )

        app.state.model_state = state
        yield

    app = FastAPI(title="mood-ml", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        # D5: o corpo padrão do FastAPI para erro de validação ({"detail":
        # [...]}) não segue {"error_code","detail"}; reescrito aqui, no único
        # ponto de serialização de erro estrutural (reforça IF-R20: nunca
        # repr(request) bruto, só a mensagem de validação do Pydantic).
        return JSONResponse(status_code=400, content={"error_code": "invalid_request", "detail": str(exc)})

    app.include_router(infer_router, prefix="/internal/v1")
    return app


app = create_app()
