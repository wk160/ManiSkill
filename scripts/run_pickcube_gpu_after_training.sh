#!/usr/bin/env bash
# Only run this after the current SwinAttUNet training has fully ended.
set -euo pipefail

SCRIPT_PATH=$BASH_SOURCE
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/maniskill_env.sh"

echo "WARNING: This GPU smoke test is only for after the current training ends."
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

if ps -p 693206 -o pid= >/dev/null 2>&1; then
    echo "Refusing GPU smoke test because protected training PID 693206 still exists." >&2
    exit 1
fi

if [[ ! -x "$MANISKILL_PYTHON" ]]; then
    echo "ManiSkill Python was not found: $MANISKILL_PYTHON" >&2
    exit 1
fi

unset CUDA_VISIBLE_DEVICES
cd "$REPO_ROOT"
exec nice -n 15 ionice -c 3 "$MANISKILL_PYTHON" - <<'PY'
import gymnasium as gym
import mani_skill.envs

env = None
try:
    env = gym.make(
        "PickCube-v1",
        num_envs=16,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=None,
        sim_backend="physx_cuda",
        # No images or GUI are requested; a CUDA render device is retained
        # solely because PickCube creates material objects at setup time.
        render_backend="cuda",
    )
    env.reset(seed=0)
    for _ in range(30):
        env.step(env.action_space.sample())
    print("GPU smoke test completed.")
finally:
    if env is not None:
        env.close()
PY
