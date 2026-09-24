"""I/O retry and atomic writes (see specs/10-common.md, CM-R01/CM-R03)."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from common.log import log

T = TypeVar("T")

IO_RETRY_BACKOFF_SECONDS = (1, 2, 4)


def with_io_retry(
    operation: str,
    func: Callable[[], T],
    run_id: str | None = None,
    *,
    logger: logging.Logger,
    error: Callable[[str], Exception],
) -> T:
    """Retries `func` on OSError after 1s/2s/4s; then raises `error(detail)`."""
    last_exc: OSError | None = None
    for attempt, backoff in enumerate((0, *IO_RETRY_BACKOFF_SECONDS), start=1):
        if backoff:
            time.sleep(backoff)
        try:
            return func()
        except OSError as exc:
            last_exc = exc
            log(logger, logging.WARNING, "io_retry", run_id=run_id, attempt=attempt, operation=operation, errno=exc.errno)
    raise error(f"{operation} failed after retries: {last_exc}")


def atomic_write(dest: Path, write_tmp: Callable[[Path], Any]) -> None:
    """Writes through `<dest>.tmp` + os.replace, so `dest` is never left partial."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    write_tmp(tmp)
    os.replace(tmp, dest)


def write_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, ensure_ascii=False)


def write_json_atomic(payload: dict[str, Any], dest: Path) -> None:
    atomic_write(dest, lambda tmp: write_json(payload, tmp))
