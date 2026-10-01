import ast
from pathlib import Path

import pytest

MOOD_ML_ROOT = Path(__file__).resolve().parent.parent.parent
EXCLUDED_DIRS = {"common", "tests", ".venv", "data", "models"}
FORBIDDEN = ("IO_RETRY_BACKOFF_SECONDS =", "time.sleep(", "lambda tmp: write_json(")

COMMON_LOG_SOURCE = (MOOD_ML_ROOT / "common" / "log.py").read_text(encoding="utf-8")
IMPORTED_FORMATTER_SOURCE = "from logging import Formatter\n\n\nclass F(Formatter):\n    pass\n"


def _subclasses_formatter(tree: ast.Module) -> bool:
    """Matches both `logging.Formatter` and a bare imported `Formatter` as a base class."""
    return any(
        (isinstance(base, ast.Attribute) and base.attr == "Formatter") or (isinstance(base, ast.Name) and base.id == "Formatter")
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for base in node.bases
    )


def _offenders(root: Path) -> list[str]:
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if EXCLUDED_DIRS & set(rel.parts):
            continue
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{rel}: {pattern}" for pattern in FORBIDDEN if pattern in text)
        if _subclasses_formatter(ast.parse(text)):
            offenders.append(f"{rel}: logging.Formatter subclass")
    return offenders


def test_ca08_infra_lives_only_in_common() -> None:
    """CM-R05: no logging.Formatter subclass, retry backoff or atomic-JSON lambda outside common/ (spec 10)."""
    assert _offenders(MOOD_ML_ROOT) == []


@pytest.mark.parametrize(
    "source", [COMMON_LOG_SOURCE, IMPORTED_FORMATTER_SOURCE], ids=["copy_of_common_log", "imported_formatter"]
)
def test_ca08_guard_flags_a_formatter_subclass(tmp_path: Path, source: str) -> None:
    copied = tmp_path / "infer" / "copied.py"
    copied.parent.mkdir()
    copied.write_text(source, encoding="utf-8")
    assert _offenders(tmp_path) != []
