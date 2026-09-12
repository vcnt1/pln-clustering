#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_DIR="$SCRIPT_DIR/mood-api"
APP_DIR="$SCRIPT_DIR/chat-app"
API_PORT="${API_PORT:-8000}"

pids=()

cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

start_back() {
    [ -x "$API_DIR/.venv/bin/uvicorn" ] || { echo "mood-api venv not found. Run ./setup-mood-api.sh first." >&2; exit 1; }
    echo "Starting mood-api on :$API_PORT"
    (cd "$API_DIR" && "$API_DIR/.venv/bin/uvicorn" app.main:app --reload --port "$API_PORT") &
    pids+=("$!")
}

start_front() {
    [ -d "$APP_DIR/node_modules" ] || { echo "chat-app dependencies not found. Run ./setup-chat-app.sh first." >&2; exit 1; }
    echo "Starting chat-app"
    (cd "$APP_DIR" && npm run dev) &
    pids+=("$!")
}

target="${1:-all}"

case "$target" in
    back) start_back ;;
    front) start_front ;;
    all) start_back; start_front ;;
    *)
        echo "Usage: $0 [back|front|all]" >&2
        exit 1
        ;;
esac

wait
