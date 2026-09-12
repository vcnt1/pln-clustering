#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/mood-ml"
VENV_DIR="$PROJECT_DIR/.venv"
REQUIRED_PYTHON="3.12"

PYTHON_BIN="$(command -v "python$REQUIRED_PYTHON" || true)"
if [ -z "$PYTHON_BIN" ]; then
    echo "python$REQUIRED_PYTHON is required but was not found on PATH." >&2
    echo "Install it (e.g. 'sudo apt install python$REQUIRED_PYTHON' or 'brew install python@$REQUIRED_PYTHON') and re-run this script." >&2
    exit 1
fi

if [ -d "$VENV_DIR" ]; then
    existing_version="$("$VENV_DIR/bin/python" --version 2>&1 | awk '{print $2}')"
    case "$existing_version" in
        "$REQUIRED_PYTHON".*) ;;
        *)
            echo "$VENV_DIR was built with Python $existing_version, not $REQUIRED_PYTHON.x. Remove it and re-run this script." >&2
            exit 1
            ;;
    esac
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

echo "mood-ml ready (Python $REQUIRED_PYTHON). Activate with: source $VENV_DIR/bin/activate"
