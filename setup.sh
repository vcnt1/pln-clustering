#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/setup-mood-api.sh"
"$SCRIPT_DIR/setup-mood-ml.sh"
"$SCRIPT_DIR/setup-chat-app.sh"

echo "All services set up."
