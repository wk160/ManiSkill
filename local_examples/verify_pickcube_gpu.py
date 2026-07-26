#!/usr/bin/env python3
"""GPU PhysX, state-only, no-renderer smoke test for ManiSkill PickCube-v1."""

import faulthandler

faulthandler.enable()

import importlib.metadata as metadata
import os
import sys
import traceback
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import sapien
import torch

import mani_skill.envs


NUM_ENVS = 16
MAX_STEPS = 100


def package_version(package_name: str) -> str:
    """Return the installed distribution version without assuming module metadata."""
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def describe_value(value: Any, path: str = "observation") -> None:
    """Recursively print structure and array/tensor metadata without copying to CPU."""
    if isinstance(value, Mapping):
        print(f"{path}: dict keys={list(value.keys())}", flush=True)
        for key, item in value.items():
            describe_value(item, f"{path}.{key}")
        return

    shape = tuple(value.shape) if hasattr(value, "shape") else None
    dtype = str(value.dtype) if hasattr(value, "dtype") else None
    device = str(value.device) if hasattr(value, "device") else None
    print(
        f"{path}: type={type(value).__name__} shape={shape} dtype={dtype} device={device}",
        flush=True,
    )


def tensor_summary(value: Any) -> tuple[float, float, float]:
    """Safely compute scalar statistics from a tensor, ndarray, or scalar."""
    if isinstance(value, torch.Tensor):
        data = value.detach().float()
        return data.mean().item(), data.min().item(), data.max().item()
    data = np.asarray(value, dtype=np.float32)
    return float(data.mean()), float(data.min()), float(data.max())


def count_true(value: Any) -> int:
    """Count truthy elements while supporting CUDA tensors and NumPy arrays."""
    if isinstance(value, torch.Tensor):
        return int(value.detach().bool().sum().item())
    return int(np.asarray(value, dtype=bool).sum())


def batch_shape(value: Any) -> tuple[int, ...] | None:
    return tuple(value.shape) if hasattr(value, "shape") else None


def combine_done(terminated: Any, truncated: Any) -> Any:
    """Return a batch done mask for either tensor or NumPy environment outputs."""
    if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
        return torch.logical_or(terminated, truncated)
    return np.logical_or(terminated, truncated)


def newly_completed(done: Any, previous_done: Any | None) -> Any:
    """Count each episode once, including later episodes after an automatic reset."""
    if previous_done is None:
        return done
    if isinstance(done, torch.Tensor):
        return torch.logical_and(done, torch.logical_not(previous_done))
    return np.logical_and(done, np.logical_not(previous_done))


def main() -> int:
    print("Python:", sys.version.replace("\n", " "), flush=True)
    print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")), flush=True)
    print("torch:", torch.__version__, flush=True)
    print("torch.version.cuda:", torch.version.cuda, flush=True)
    print("torch.cuda.is_available:", torch.cuda.is_available(), flush=True)
    print("torch.cuda.device_count:", torch.cuda.device_count(), flush=True)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("torch CUDA device 0:", torch.cuda.get_device_name(0), flush=True)
    print("ManiSkill:", package_version("mani_skill"), flush=True)
    print("SAPIEN:", package_version("sapien"), flush=True)
    print("Gymnasium:", package_version("gymnasium"), flush=True)
    print("NumPy:", np.__version__, flush=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        print("Refusing to run: CUDA_VISIBLE_DEVICES must not be an empty string.", flush=True)
        return 2
    if not torch.cuda.is_available():
        print("Refusing to run: PyTorch CUDA is unavailable.", flush=True)
        return 2

    env = None
    status = 1
    try:
        print("creating environment", flush=True)
        env = gym.make(
            "PickCube-v1",
            num_envs=NUM_ENVS,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            sim_backend="physx_cuda",
            render_backend="none",
            render_mode=None,
        )
        print("environment created", flush=True)
        print("simulation backend:", env.unwrapped.backend.sim_backend, flush=True)
        print("render backend:", env.unwrapped.backend.render_backend, flush=True)
        print("environment num_envs:", env.unwrapped.num_envs, flush=True)
        if env.unwrapped.backend.sim_backend != "physx_cuda":
            raise RuntimeError("Expected sim_backend=physx_cuda")
        if env.unwrapped.backend.render_backend != "none":
            raise RuntimeError("Expected render_backend=none")
        if env.unwrapped.num_envs != NUM_ENVS:
            raise RuntimeError(f"Expected num_envs={NUM_ENVS}")

        env.action_space.seed(0)
        observation, info = env.reset(seed=0)
        print("reset passed", flush=True)
        print("observation_space:", env.observation_space, flush=True)
        print("action_space:", env.action_space, flush=True)
        describe_value(observation)
        print("reset info keys:", sorted(info.keys()), flush=True)

        action = env.action_space.sample()
        print(
            "action_space.sample:"
            f" type={type(action).__name__} shape={batch_shape(action)}"
            f" dtype={getattr(action, 'dtype', None)}",
            flush=True,
        )
        action_shape = batch_shape(action)
        if action_shape is None or action_shape[0] != NUM_ENVS:
            raise RuntimeError(
                "The vectorized action_space sample does not have the expected "
                f"leading batch dimension {NUM_ENVS}: {action_shape}"
            )

        print("expected reward batch shape:", (NUM_ENVS,), flush=True)
        print("expected terminated batch shape:", (NUM_ENVS,), flush=True)
        print("expected truncated batch shape:", (NUM_ENVS,), flush=True)

        completed_episodes = 0
        previous_done = None
        for step_index in range(1, MAX_STEPS + 1):
            # The vectorized action space already samples one action per environment.
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = combine_done(terminated, truncated)
            completed_episodes += count_true(newly_completed(done, previous_done))
            previous_done = done

            if step_index % 10 == 0:
                reward_mean, reward_min, reward_max = tensor_summary(reward)
                success = info.get("success") if isinstance(info, Mapping) else None
                success_count = count_true(success) if success is not None else "unavailable"
                success_rate = (
                    tensor_summary(success)[0] if success is not None else "unavailable"
                )
                print(
                    "step=",
                    step_index,
                    "reward_mean=",
                    reward_mean,
                    "reward_min=",
                    reward_min,
                    "reward_max=",
                    reward_max,
                    "reward_shape=",
                    batch_shape(reward),
                    "reward_device=",
                    getattr(reward, "device", None),
                    "terminated_shape=",
                    batch_shape(terminated),
                    "terminated_count=",
                    count_true(terminated),
                    "truncated_shape=",
                    batch_shape(truncated),
                    "truncated_count=",
                    count_true(truncated),
                    "success_count=",
                    success_count,
                    "success_rate=",
                    success_rate,
                    "completed_episodes=",
                    completed_episodes,
                    flush=True,
                )

        print(f"{MAX_STEPS} steps passed", flush=True)
        status = 0
    except Exception:
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
                print("environment closed", flush=True)
            except Exception:
                traceback.print_exc()
                status = 1

    if status == 0:
        print("GPU PhysX smoke test passed", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
