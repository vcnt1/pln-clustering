from pathlib import Path

MOOD_ML_ROOT = Path(__file__).resolve().parent.parent.parent


def test_no_duckdb_reference_in_ingest_and_mask() -> None:
    """P1: mood-ml never opens the DuckDB file (constitution.md P1)."""
    targets = [MOOD_ML_ROOT / "ingest", MOOD_ML_ROOT / "transform" / "mask.py"]
    offenders = []
    for target in targets:
        files = [target] if target.is_file() else sorted(target.rglob("*.py"))
        for path in files:
            text = path.read_text(encoding="utf-8")
            if "duckdb" in text.lower():
                offenders.append(path)
    assert offenders == []
