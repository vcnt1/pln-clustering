"""CA-06 (spec 11): os módulos de A nunca importam `torch`, mesmo com `torch`
instalado no ambiente (`requirements-embeddings.txt`) — só quem usa
`algorithm=embeddings-ft` paga esse custo, via imports *lazy* dentro de
função (FT-R02). Roda em subprocess com `sys.modules` limpo: o processo da
suíte principal já pode ter `torch` importado por outros arquivos de teste
marcados `embeddings`, o que mascararia uma regressão aqui."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_SCRIPT = """
import sys
import train.train
import evaluate.metrics
import registry.registry
import infer.predict
import transform.features
import pipeline
import main

assert "torch" not in sys.modules, sorted(m for m in sys.modules if "torch" in m)
print("OK")
"""


def test_core_modules_do_not_import_torch_as_a_side_effect() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
