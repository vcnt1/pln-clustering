import json
import logging
from pathlib import Path

import pytest

import common.io as common_io
from common.errors import PipelineError
from common.io import atomic_write, with_io_retry, write_json
from common.log import configure_logging, log


class _FakeIOError(PipelineError):
    pass


def _error(detail: str) -> _FakeIOError:
    return _FakeIOError("FAKE_IO_FAILED", detail)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(common_io.time, "sleep", recorded.append)
    return recorded


def _json_logger() -> logging.Logger:
    # Built inside the test body: pytest swaps sys.stderr between setup and call.
    logger = logging.getLogger("tests.common")
    configure_logging(logger, "json", "INFO")
    return logger


def _flaky(failures: int):
    calls = {"n": 0}

    def _func() -> str:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise OSError(13, "locked")
        return "ok"

    return _func, calls


# ---------------------------------------------------------------------------
# with_io_retry — CM-R01
# ---------------------------------------------------------------------------


def test_ca01_retry_succeeds_after_transient_failures(sleeps: list[float]) -> None:
    func, calls = _flaky(failures=2)
    assert with_io_retry("read_x", func, logger=_json_logger(), error=_error) == "ok"
    assert calls["n"] == 3
    assert sleeps == [1, 2]


def test_ca02_retry_raises_layer_error_after_exhausting_attempts(
    sleeps: list[float],
) -> None:
    func, calls = _flaky(failures=99)
    with pytest.raises(_FakeIOError) as exc:
        with_io_retry("read_x", func, logger=_json_logger(), error=_error)
    assert exc.value.code == "FAKE_IO_FAILED"
    assert "read_x failed after retries" in exc.value.detail
    assert calls["n"] == 4
    assert sleeps == [1, 2, 4]


def test_ca03_non_os_error_propagates_without_retry(sleeps: list[float]) -> None:
    def _boom() -> None:
        raise ValueError("not I/O")

    with pytest.raises(ValueError):
        with_io_retry("read_x", _boom, logger=_json_logger(), error=_error)
    assert sleeps == []


def test_ca04_io_retry_is_a_flat_json_line(
    sleeps: list[float], capsys: pytest.CaptureFixture[str]
) -> None:
    func, _ = _flaky(failures=1)
    with_io_retry("read_x", func, run_id="r1", logger=_json_logger(), error=_error)

    line = capsys.readouterr().err.strip().splitlines()[0]
    record = json.loads(line)
    assert record["event"] == "io_retry"
    assert record["run_id"] == "r1"
    assert record["operation"] == "read_x"
    assert record["attempt"] == 1
    assert {"ts", "level"} <= record.keys()


# ---------------------------------------------------------------------------
# atomic_write — CM-R03
# ---------------------------------------------------------------------------


def test_ca05_atomic_write_replaces_and_leaves_no_tmp(tmp_path: Path) -> None:
    dest = tmp_path / "nested" / "report.json"
    atomic_write(dest, lambda tmp: write_json({"b": 1, "a": "ç"}, tmp))
    atomic_write(dest, lambda tmp: write_json({"a": 2}, tmp))

    assert json.loads(dest.read_text(encoding="utf-8")) == {"a": 2}
    assert list(dest.parent.glob("*.tmp")) == []


def test_write_json_keeps_the_existing_on_disk_format(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    write_json({"b": 1, "a": "ç"}, path)
    assert path.read_text(encoding="utf-8") == '{\n  "a": "ç",\n  "b": 1\n}'


# ---------------------------------------------------------------------------
# log / configure_logging — CM-R02
# ---------------------------------------------------------------------------


def test_ca06_log_omits_run_id_when_none(capsys: pytest.CaptureFixture[str]) -> None:
    json_logger = _json_logger()
    log(json_logger, logging.INFO, "without_run")
    log(json_logger, logging.INFO, "with_run", run_id="x", n=1)

    first, second = (json.loads(line) for line in capsys.readouterr().err.strip().splitlines())
    assert "run_id" not in first
    assert second["run_id"] == "x"
    assert second["n"] == 1


def test_ca07_configure_logging_twice_keeps_a_single_handler() -> None:
    logger = logging.getLogger("tests.common.handlers")
    configure_logging(logger, "json", "INFO")
    configure_logging(logger, "text", "WARNING")
    assert len(logger.handlers) == 1
    assert logger.level == logging.WARNING
