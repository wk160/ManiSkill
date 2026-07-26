#!/usr/bin/env bash
# Stage 04 PPO smoke test，不代表策略收敛，也不是正式训练。
set -euo pipefail

SCRIPT_PATH=${BASH_SOURCE[0]}
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON=/home/ubuntu/anaconda/envs/maniskill/bin/python
PPO_SCRIPT="$REPO_ROOT/examples/baselines/ppo/ppo.py"
OUTPUT_ROOT=/home/ubuntu/wk/maniskill_runs/ppo_smoke

echo "Stage 04 PPO smoke test，不代表策略收敛，也不是正式训练。"
echo "将使用 PickCube-v1 state、16 个训练环境、64-step rollout 和无渲染 GPU PhysX。"
echo "不会上传 WandB、不会录制视频、不会启动 GUI；外部 GPU compute 进程允许共存。"

if [[ ! -x "$PYTHON" ]]; then
    echo "ManiSkill Python was not found: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$PPO_SCRIPT" ]]; then
    echo "Official PPO entrypoint was not found: $PPO_SCRIPT" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; refusing PPO smoke test." >&2
    exit 1
fi

echo "GPU compute processes before PPO:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')"
echo "GPU free memory before PPO: ${free_mib} MiB"
if [[ "$free_mib" -lt 10240 ]]; then
    echo "Refusing PPO smoke test: free GPU memory is below the 10 GiB safety line." >&2
    exit 1
fi

if [[ "$free_mib" -ge 12288 ]]; then
    total_timesteps=65536
else
    total_timesteps=32768
fi

mkdir -p "$OUTPUT_ROOT"
run_id="stage04_$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$OUTPUT_ROOT/$run_id"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing smoke-test directory: $RUN_DIR" >&2
    exit 1
fi
mkdir "$RUN_DIR"

export CUDA_VISIBLE_DEVICES=0
export MS_ASSET_DIR=/home/ubuntu/wk/maniskill_data
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export PYTHONFAULTHANDLER=1

# ppo.py writes TensorBoard events and checkpoints to runs/<exp-name> relative
# to the working directory, so both remain below this unique smoke-test directory.
cd "$RUN_DIR"
set +e
nice -n 15 ionice -c 3 "$PYTHON" -X faulthandler "$PPO_SCRIPT" \
    --exp-name pickcube_state_stage04 \
    --env-id PickCube-v1 \
    --cuda \
    --no-track \
    --no-capture-video \
    --save-model \
    --render-mode none \
    --render-backend none \
    --sim-backend physx_cuda \
    --num-envs 16 \
    --num-eval-envs 1 \
    --num-steps 64 \
    --num-eval-steps 64 \
    --total-timesteps "$total_timesteps" \
    --num-minibatches 4 \
    --update-epochs 4 \
    --eval-freq 1 \
    --control-mode pd_joint_delta_pos \
    2>&1 | tee "$RUN_DIR/ppo_smoke.log"
ppo_status=${PIPESTATUS[0]}
set -e

echo "PPO smoke test exit code: $ppo_status"
echo "Stage 04 run directory: $RUN_DIR"
echo "TensorBoard/checkpoint (if successful): $RUN_DIR/runs/pickcube_state_stage04"
exit "$ppo_status"
