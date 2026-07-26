# PickCube PPO 下一阶段说明

## 当前状态

- GPU PhysX state-only smoke test 已在 `PickCube-v1`、16 个环境、100 step 下通过。
- Stage 04 PPO smoke 已于 2026-07-24 22:47--22:49 CST 完成：`scripts/run_pickcube_ppo_smoke.sh` 创建时间戳子目录且不覆盖旧结果。
- 该 PPO smoke 不是已训练好的模型，也不能作为收敛、成功率或独占 GPU 性能结论。

## 官方 PPO 实现

- 当前入口：`examples/baselines/ppo/ppo.py`。
- Actor：state 输入经 `42 -> 256 -> 256 -> 256 -> 8` 的 Tanh MLP，外加可学习高斯策略 log-std。
- Critic：state 输入经 `42 -> 256 -> 256 -> 256 -> 1` 的 Tanh MLP。
- 数据流：`state observation -> policy -> action -> GPU PhysX -> reward / termination -> rollout buffer -> PPO update`。
- 当前源码固定 `obs_mode="state"`，并新增了向后兼容的 `--render-mode`、`--render-backend` 和 `--sim-backend`。默认仍为 `rgb_array`、`gpu`、`physx_cuda`；`--render-mode none` 会转换为 Python `None`，`--render-backend none` 会转换为 ManiSkill 3.0.1 所需字符串 `"none"`。

## 已准备的 smoke 规模

Stage 04 使用当前 `ppo.py --help` 认可的参数：16 个训练环境、1 个评估环境、64-step rollout、65536 个训练 timesteps、4 个 minibatch、4 个 update epoch。rollout batch 是 1024，共完成 64 个 PPO update 迭代；训练耗时约 137 秒。它与外部 SwinAttUNet evaluation PID 746284 并行，故 GPU 总显存峰值 8893 MiB 和总利用率峰值 76% 不能完全归因于 PPO；PPO 进程自身峰值为 3768 MiB。

脚本显式启用 CUDA、设置 `CUDA_VISIBLE_DEVICES=0`，设置 `--no-track` 禁止 WandB 上传，设置 `--no-capture-video` 禁止视频，并显式请求 `render_mode=None`、`render_backend="none"` 与 `sim_backend=physx_cuda`。它允许外部 compute process 共存，但在可用显存低于 10 GiB 时拒绝运行；每次创建时间戳子目录，绝不覆盖或删除旧输出。Stage 04 生成了可读 final checkpoint 和 TensorBoard event，所有记录 loss 标量均为有限值。

官方默认规模是 512 训练环境、8 评估环境、50-step rollout、32 minibatch、4 个 update epoch 和 10,000,000 timesteps。正式训练前应先单独确认目标任务、学习率、rollout、评估频率、磁盘预算及离线指标；不要直接把 smoke 配置扩大为正式训练配置。

## 执行前的人工确认

仅在明确同意实际启动 PPO、并确认可用显存不低于 10 GiB 后运行（外部 compute process 可以共存）：

```bash
/bin/bash /home/ubuntu/wk/projects/ManiSkill/scripts/run_pickcube_ppo_smoke.sh
```

显存监控：

```bash
watch -n 2 nvidia-smi
```

TensorBoard（成功运行后）：

```bash
/home/ubuntu/anaconda/envs/maniskill/bin/tensorboard \
  --logdir /home/ubuntu/wk/maniskill_runs/ppo_smoke \
  --port 6006
```

可在确认后使用 tmux 保持会话：

```bash
tmux new -s maniskill-ppo-smoke
/bin/bash /home/ubuntu/wk/projects/ManiSkill/scripts/run_pickcube_ppo_smoke.sh
```

安全停止：在该 tmux 窗口中按一次 `Ctrl-C`，等待 `ppo.py` 的 close 路径和 shell 返回；不要使用 `kill -9`。之后检查 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits` 为空。

Stage 04 已成功生成的 checkpoint 位于：

```text
/home/ubuntu/wk/maniskill_runs/ppo_smoke/stage04_20260724_224718/runs/pickcube_state_stage04/final_ckpt.pt
```

日志和 TensorBoard event 文件也在同一目录。运行后只剩预先存在的外部 PID 746284，未发现本次 PPO 残留。详细摘要为 `deployment_logs/PPO_SMOKE_STAGE04_SUMMARY.md`；该结果仅证明 PPO 训练链路连通，不代表策略已经收敛，也不代表正式训练已经开始。
