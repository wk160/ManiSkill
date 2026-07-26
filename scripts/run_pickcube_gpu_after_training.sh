#!/usr/bin/env bash
# Only run this after all unrelated GPU compute jobs have fully ended.
set -euo pipefail

SCRIPT_PATH=$BASH_SOURCE
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/maniskill_env.sh"

echo "WARNING: This GPU smoke test is only for after unrelated training ends."
echo "It refuses to run while any GPU compute process is present."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing GPU smoke test." >&2
    exit 1
fi

gpu_processes="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "$gpu_processes" ]]; then
    echo "Refusing GPU smoke test because GPU compute processes are active:" >&2
    echo "$gpu_processes" >&2
    exit 1
fi

if [[ ! -x "$MANISKILL_PYTHON" ]]; then
    echo "ManiSkill Python was not found: $MANISKILL_PYTHON" >&2
    exit 1
fi

cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONFAULTHANDLER=1

set +e
nice -n 15 ionice -c 3 "$MANISKILL_PYTHON" -X faulthandler \
    "$REPO_ROOT/local_examples/verify_pickcube_gpu.py" 2>&1 | \
    tee "${REPO_ROOT}/deployment_logs/verify_pickcube_gpu.log"
gpu_test_status=${PIPESTATUS[0]}
set -e

echo "GPU smoke test exit code: $gpu_test_status"
exit "$gpu_test_status"
