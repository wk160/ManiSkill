#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$BASH_SOURCE
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/maniskill_env.sh"

if [[ ! -x "$MANISKILL_PYTHON" ]]; then
    echo "ManiSkill Python was not found: $MANISKILL_PYTHON" >&2
    exit 1
fi

cd "$REPO_ROOT"
exec nice -n 15 ionice -c 3 "$MANISKILL_PYTHON" "$REPO_ROOT/local_examples/verify_pickcube_cpu.py"
