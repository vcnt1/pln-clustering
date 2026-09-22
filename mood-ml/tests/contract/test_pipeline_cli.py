import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import ingest.validate as ingest_validate
import pipeline
from tests.conftest import make_row, write_corpus


def _load_fixture_rows(fixture_path: Path) -> list[dict]:
    return [json.loads(line) for line in fixture_path.read_text(encoding="utf-8").splitlines() if line]


def _stage_corpus(directory: Path, fixture_path: Path, corpus_id: str) -> Path:
    """Copies a committed fixture (already carrying `corpus_id` in every
    row) into <directory>/raw/synthetic/<corpus_id>.jsonl, satisfying
    IG-R02/IG-R03 — the fixtures under tests/fixtures/ are not in that
    layout by design (raw fixtures, staged by whichever test needs them)."""
    rows = _load_fixture_rows(fixture_path)
    return write_corpus(directory, corpus_id, rows)


MEDIUM_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_medium.jsonl"
SMALL_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample.jsonl"


# ---------------------------------------------------------------------------
# CA-01 — subcomandos individuais são equivalentes ao módulo chamado direto
# ---------------------------------------------------------------------------


def test_ingest_subcommand_matches_the_module_called_directly(tmp_path: Path) -> None:
    corpus_path = _stage_corpus(tmp_path, SMALL_FIXTURE, "syn-2026-01-01-a")

    direct_report_dir = tmp_path / "reports_direct"
    direct_code = ingest_validate.main([str(corpus_path), "--report-dir", str(direct_report_dir), "--skip-composition"])

    piped_report_dir = tmp_path / "reports_pipeline"
    piped_code = pipeline.main(
        ["ingest", str(corpus_path), "--report-dir", str(piped_report_dir), "--skip-composition"]
    )

    assert direct_code == piped_code

    direct_report = json.loads((direct_report_dir / "syn-2026-01-01-a" / "validation_report.json").read_text(encoding="utf-8"))
    piped_report = json.loads((piped_report_dir / "syn-2026-01-01-a" / "validation_report.json").read_text(encoding="utf-8"))
    # "run"/"source" carregam run_id e timestamps que variam entre execuções
    # independentes — a equivalência que importa é o resultado, não o envelope.
    assert direct_report["status"] == piped_report["status"]
    assert sorted(e["code"] for e in direct_report["errors"]) == sorted(e["code"] for e in piped_report["errors"])
    assert direct_report["counts"] == piped_report["counts"]


# ---------------------------------------------------------------------------
# CA-02 / CA-06 / CA-07 — `all` de ponta a ponta
# ---------------------------------------------------------------------------


def _all_args(tmp_path: Path, corpus_path: Path, **overrides: str) -> list[str]:
    args = [
        "all",
        str(corpus_path),
        "--skip-composition",
        "--report-dir", str(tmp_path / "reports"),
        "--labels-dir", str(tmp_path / "labels"),
        "--datasets-dir", str(tmp_path / "datasets"),
        "--staging-dir", str(tmp_path / "staging"),
        "--models-dir", str(tmp_path / "models"),
    ]
    for flag, value in overrides.items():
        args.extend([f"--{flag.replace('_', '-')}", value])
    return args


def test_all_runs_end_to_end_and_registers_a_model(tmp_path: Path) -> None:
    corpus_path = _stage_corpus(tmp_path, MEDIUM_FIXTURE, "syn-2026-01-02-a")

    code = pipeline.main(_all_args(tmp_path, corpus_path))
    assert code == 0

    today = datetime.now(timezone.utc).date()
    expected_dataset_dir = tmp_path / "datasets" / f"ds-{today.year:04d}-{today.month:02d}-{today.day:02d}-a"
    assert expected_dataset_dir.exists()

    model_dirs = list((tmp_path / "models").glob("mood-*"))
    assert len(model_dirs) == 1
    assert (model_dirs[0] / "model.joblib").exists()
    assert (model_dirs[0] / "manifest.json").exists()

    # active.json nunca é criado por `all` (D4/PL-R08) — promote é sempre manual.
    assert not (tmp_path / "models" / "active.json").exists()

    # CA-06 (fingerprint correto) é implícita aqui: se _recompute_fingerprint
    # estivesse errado, o portão de registry.register rejeitaria (nenhum
    # staging bateria com aquele fingerprint) e `all` teria falhado com exit 3.
    manifest = json.loads((model_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    staging_dir = tmp_path / "staging" / expected_dataset_dir.name / "tfidf-ridge" / manifest["training_fingerprint"]
    train_manifest = json.loads((staging_dir / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_fingerprint"] == train_manifest["fingerprint"]


# ---------------------------------------------------------------------------
# CA-03 — dataset_id incrementa entre corridas do mesmo dia
# ---------------------------------------------------------------------------


def test_all_mints_distinct_dataset_ids_on_repeated_runs(tmp_path: Path) -> None:
    corpus_path = _stage_corpus(tmp_path, MEDIUM_FIXTURE, "syn-2026-01-02-a")

    assert pipeline.main(_all_args(tmp_path, corpus_path)) == 0
    assert pipeline.main(_all_args(tmp_path, corpus_path)) == 0

    dataset_dirs = sorted(d.name for d in (tmp_path / "datasets").iterdir())
    today = datetime.now(timezone.utc).date()
    prefix = f"ds-{today.year:04d}-{today.month:02d}-{today.day:02d}-"
    assert dataset_dirs == [f"{prefix}a", f"{prefix}b"]

    # O mesmo corpus, com o split de seed fixa (configs/pipeline.yaml), produz
    # o mesmo particionamento train/validation/test byte-a-byte nas duas
    # corridas -> mesmo fingerprint -> a idempotência de registry.register já
    # existente reconhece o mesmo candidato e reusa o model_version, em vez de
    # mintar um segundo. dataset_id diferente (PL-R05) não implica candidato
    # diferente quando a entrada é idêntica -- comportamento correto, não bug.
    model_dirs = sorted(d.name for d in (tmp_path / "models").glob("mood-*"))
    assert len(model_dirs) == 1


# ---------------------------------------------------------------------------
# CA-04 — corpus inválido para no passo ingest, exit propagado sem tradução
# ---------------------------------------------------------------------------


def test_all_stops_at_ingest_and_propagates_the_same_exit_code(tmp_path: Path) -> None:
    bad_row = make_row(role="customer", generated_label=0.3)  # DC_R07_LABEL_MISSING_OR_INVALID
    bad_corpus_path = write_corpus(tmp_path, "syn-2026-01-01-a", [bad_row])

    standalone_code = ingest_validate.main(
        [str(bad_corpus_path), "--report-dir", str(tmp_path / "reports_standalone"), "--skip-composition"]
    )
    assert standalone_code != 0

    all_code = pipeline.main(_all_args(tmp_path, bad_corpus_path))
    assert all_code == standalone_code
    assert not (tmp_path / "datasets").exists()


# ---------------------------------------------------------------------------
# CA-05 — quality gate reprovado interrompe antes de register
# ---------------------------------------------------------------------------


def test_all_stops_before_register_when_evaluate_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path = _stage_corpus(tmp_path, SMALL_FIXTURE, "syn-2026-01-01-a")

    register_calls = []
    monkeypatch.setattr(pipeline.evaluate_metrics, "main", lambda argv: 5)
    monkeypatch.setattr(pipeline.registry_registry, "main", lambda argv: register_calls.append(argv) or 0)

    code = pipeline.main(_all_args(tmp_path, corpus_path))

    assert code == 5
    assert register_calls == []
    assert not list((tmp_path / "models").glob("mood-*"))


# ---------------------------------------------------------------------------
# _mint_dataset_id determinism smoke-check via the real `all` flow
# ---------------------------------------------------------------------------


def test_mint_dataset_id_used_by_all_matches_todays_date(tmp_path: Path) -> None:
    corpus_path = _stage_corpus(tmp_path, MEDIUM_FIXTURE, "syn-2026-01-02-a")
    assert pipeline.main(_all_args(tmp_path, corpus_path)) == 0

    dataset_dir = next((tmp_path / "datasets").iterdir())
    today = datetime.now(timezone.utc).date()
    assert dataset_dir.name == f"ds-{today.year:04d}-{today.month:02d}-{today.day:02d}-a"


# ---------------------------------------------------------------------------
# CA-09 — P1
# ---------------------------------------------------------------------------


def test_no_duckdb_reference_in_pipeline() -> None:
    module_path = Path(__file__).resolve().parent.parent.parent / "pipeline.py"
    assert "duckdb" not in module_path.read_text(encoding="utf-8").lower()
