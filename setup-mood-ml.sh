#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/mood-ml"
VENV_DIR="$PROJECT_DIR/.venv"

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }

[ -d "$VENV_DIR" ] || python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "mood-ml ready. Activate with: source $VENV_DIR/bin/activate"
