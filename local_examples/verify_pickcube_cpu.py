#!/usr/bin/env python3
"""CPU PhysX, no-GUI/no-image smoke test for ManiSkill PickCube-v1."""

import os

# Set project-local paths before importing torch, SAPIEN, or ManiSkill.
# CUDA visibility is deliberately inherited from the caller: CPU PhysX still
# needs a GPU-visible process for this SAPIEN version to construct task
# materials safely.
os.environ.setdefault("MS_ASSET_DIR", "/home/ubuntu/wk/maniskill_data")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/home/ubuntu/wk/maniskill_cache/matplotlib")

import importlib.metadata as metadata
import sys
import traceback
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

import mani_skill.envs


def package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def describe_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: describe_observation(item) for key, item in value.items()}
    return {
        "type": type(value).__name__,
        "shape": tuple(value.shape) if hasattr(value, "shape") else None,
        "dtype": str(value.dtype) if hasattr(value, "dtype") else None,
    }


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch CUDA visible:", torch.cuda.is_available())
    print("ManiSkill:", package_version("mani_skill"))
    print("SAPIEN:", package_version("sapien"))
    print("Gymnasium:", package_version("gymnasium"))
    print("NumPy:", np.__version__)
    print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

    env = None
    try:
        env = gym.make(
            "PickCube-v1",
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cpu",
            render_backend="none",
            render_mode=None,
        )
        print("simulation backend:", env.unwrapped.backend.sim_backend)
        print("render backend:", env.unwrapped.backend.render_backend)
        print("observation_space:", env.observation_space)
        print("action_space:", env.action_space)

        seed = 0
        env.action_space.seed(seed)
        observation, info = env.reset(seed=seed)
        print("first observation:", describe_observation(observation))
        print("reset info keys:", sorted(info.keys()))

        reward = terminated = truncated = info = None
        for step_index in range(30):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if (step_index + 1) % 10 == 0:
                print(
                    "step",
                    step_index + 1,
                    "reward=",
                    reward,
                    "terminated=",
                    terminated,
                    "truncated=",
                    truncated,
                )

        print("final reward:", reward)
        print("final terminated:", terminated)
        print("final truncated:", truncated)
        print("final info keys:", sorted(info.keys()))
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
            print("environment closed")


if __name__ == "__main__":
    raise SystemExit(main())
