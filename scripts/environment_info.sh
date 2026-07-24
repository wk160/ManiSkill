#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH=$BASH_SOURCE
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/maniskill_env.sh"

echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "Repository: $REPO_ROOT"
echo "Git commit: $(git -C "$REPO_ROOT" log -1 --oneline)"
echo "Git branch: $(git -C "$REPO_ROOT" branch --show-current)"
echo "Git remotes:"
git -C "$REPO_ROOT" remote -v
echo "Python packages:"
"$MANISKILL_PYTHON" - <<'PY'
import importlib.metadata as metadata
import sys
import torch

print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
for package in ("sapien", "mani_skill", "gymnasium"):
    print(f"{package}:", metadata.version(package))
PY
echo "nvidia-smi:"
nvidia-smi || true
echo "Root filesystem:"
df -h /
echo "Asset directory size:"
du -sh "$MS_ASSET_DIR" 2>/dev/null || true
echo "Conda environment size:"
du -sh /home/ubuntu/anaconda/envs/maniskill 2>/dev/null || true
