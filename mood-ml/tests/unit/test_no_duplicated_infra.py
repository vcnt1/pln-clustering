from pathlib import Path

MOOD_ML_ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_DIRS = {"common", "tests", ".venv", "data", "models"}
FORBIDDEN = ("class _JsonFormatter", "class _TextFormatter", "IO_RETRY_BACKOFF_SECONDS =", "time.sleep(")


def test_ca08_infra_lives_only_in_common() -> None:
    """CM-R05: retry backoff and log formatters are never copied outside common/ (spec 10)."""
    offenders = []
    for path in sorted(MOOD_ML_ROOT.rglob("*.py")):
        if EXCLUDED_DIRS & set(path.relative_to(MOOD_ML_ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path.relative_to(MOOD_ML_ROOT)}: {pattern}" for pattern in FORBIDDEN if pattern in text)
    assert offenders == []
