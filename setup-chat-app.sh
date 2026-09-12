#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/chat-app"

command -v npm >/dev/null 2>&1 || { echo "npm is required" >&2; exit 1; }

(cd "$PROJECT_DIR" && npm install)

echo "chat-app ready. Run with: cd $PROJECT_DIR && npm run dev"
