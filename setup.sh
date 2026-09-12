#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIRED_PYTHON="3.12"

ensure_brew() {
    if command -v brew >/dev/null 2>&1; then
        echo "brew found: $(command -v brew)"
        return
    fi

    echo "brew not found, installing Homebrew..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
        echo "Homebrew install failed. Install it manually: https://brew.sh" >&2
        exit 1
    }

    for brew_bin in /home/linuxbrew/.linuxbrew/bin/brew "$HOME/.linuxbrew/bin/brew" /opt/homebrew/bin/brew /usr/local/bin/brew; do
        if [ -x "$brew_bin" ]; then
            eval "$("$brew_bin" shellenv)"
            break
        fi
    done

    command -v brew >/dev/null 2>&1 || { echo "brew still not found after install attempt. Install it manually: https://brew.sh" >&2; exit 1; }
}

ensure_python() {
    if command -v "python$REQUIRED_PYTHON" >/dev/null 2>&1; then
        echo "python$REQUIRED_PYTHON found: $(command -v "python$REQUIRED_PYTHON")"
        return
    fi

    echo "python$REQUIRED_PYTHON not found, attempting install..."
    if command -v brew >/dev/null 2>&1; then
        brew install "python@$REQUIRED_PYTHON"
    elif command -v apt-get >/dev/null 2>&1; then
        echo "Needs sudo, run this yourself then re-run ./setup.sh:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y python$REQUIRED_PYTHON python$REQUIRED_PYTHON-venv" >&2
        exit 1
    else
        echo "Install Python $REQUIRED_PYTHON manually and re-run ./setup.sh." >&2
        exit 1
    fi

    command -v "python$REQUIRED_PYTHON" >/dev/null 2>&1 || { echo "python$REQUIRED_PYTHON still not found after install attempt." >&2; exit 1; }
}

ensure_node() {
    if command -v npm >/dev/null 2>&1; then
        echo "npm found: $(command -v npm)"
        return
    fi

    echo "npm not found, attempting install..."
    if command -v brew >/dev/null 2>&1; then
        brew install node
    elif command -v apt-get >/dev/null 2>&1; then
        echo "Needs sudo, run this yourself then re-run ./setup.sh:" >&2
        echo "  sudo apt-get update && sudo apt-get install -y nodejs npm" >&2
        exit 1
    else
        echo "Install Node.js/npm manually and re-run ./setup.sh." >&2
        exit 1
    fi

    command -v npm >/dev/null 2>&1 || { echo "npm still not found after install attempt." >&2; exit 1; }
}

ensure_brew
ensure_python
ensure_node

"$SCRIPT_DIR/setup-mood-api.sh"
"$SCRIPT_DIR/setup-mood-ml.sh"
"$SCRIPT_DIR/setup-chat-app.sh"

echo "All services set up."
