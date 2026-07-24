#!/usr/bin/env bash
# Source this file from ManiSkill helper scripts. It does not activate Conda.

SCRIPT_PATH=$BASH_SOURCE
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
export MANISKILL_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export MS_ASSET_DIR=/home/ubuntu/wk/maniskill_data
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
export PIP_CACHE_DIR=/home/ubuntu/wk/maniskill_cache/pip
export TORCH_HOME=/home/ubuntu/wk/maniskill_cache/torch
export TMPDIR=/home/ubuntu/wk/maniskill_tmp
export MANISKILL_PYTHON=/home/ubuntu/anaconda/envs/maniskill/bin/python
