import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

BASE_ROW = {
    "corpus_id": "syn-2026-01-01-a",
    "conversation_id": "syn-conv-0001",
    "customer_id": "syn-cust-0001",
    "message_id": "syn-msg-000001",
    "role": "customer",
    "text": "Qual o horario do check-out?",
    "sent_at": "2026-01-01T10:00:00Z",
    "persona": "objetivo",
    "generated_label": 0.0,
}


def make_row(**overrides: object) -> dict:
    row = dict(BASE_ROW)
    row.update(overrides)
    return row


def write_corpus(directory: Path, corpus_id: str, rows: list[dict]) -> Path:
    """Writes `rows` as a .jsonl corpus under <directory>/raw/synthetic/<corpus_id>.jsonl,
    satisfying IG-R02 (path must be under data/raw/synthetic/)."""
    target_dir = directory / "raw" / "synthetic"
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{corpus_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def corpus_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_jsonl_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "sample.jsonl"
