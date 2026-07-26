# ManiSkill 3 Ubuntu 隔离部署记录

部署日期：2026-07-22  
主机：ubuntu-server  
仓库：/home/ubuntu/wk/projects/ManiSkill  
Git 分支：feature/upper-limb-rl  
Git HEAD：1fcfcd4 chore(deploy): add ManiSkill setup and CPU PhysX verification

## 已部署环境

- GPU：NVIDIA GeForce RTX 4090
- NVIDIA Driver：550.144.03
- nvidia-smi CUDA Version：12.4
- Conda 环境：/home/ubuntu/anaconda/envs/maniskill
- Python：3.11.15
- PyTorch：2.5.1+cu121
- PyTorch CUDA wheel：cu121
- PyTorch 安装索引：https://download.pytorch.org/whl/cu121
- ManiSkill：3.0.1，使用当前源码 editable 安装
- SAPIEN：3.0.3
- Gymnasium：1.3.0
- NumPy：2.4.6
- SciPy：1.17.1
- TensorBoard：2.21.0

选择 cu121 是因为现有 swinattunet 训练环境已在同一台服务器上使用 PyTorch 2.5.1+cu121；NVIDIA 550.144.03 驱动支持该 wheel 运行时。没有安装系统 CUDA Toolkit、torchvision 或 torchaudio。

## 隔离目录

- 资产：/home/ubuntu/wk/maniskill_data
- 运行输出：/home/ubuntu/wk/maniskill_runs
- 缓存：/home/ubuntu/wk/maniskill_cache
- 临时文件：/home/ubuntu/wk/maniskill_tmp
- 本仓库部署日志：deployment_logs，受 Git 忽略

项目脚本 scripts/maniskill_env.sh 只为 ManiSkill 脚本设置 MS_ASSET_DIR、MS_SKIP_ASSET_DOWNLOAD_PROMPT、PIP_CACHE_DIR、TORCH_HOME、TMPDIR 和 MANISKILL_PYTHON。它不会激活或改动 swinattunet，也不会写入系统 profile、bashrc 或 environment。

## CPU 验证状态

验证命令：

~~~bash
CUDA_VISIBLE_DEVICES=0 /home/ubuntu/wk/projects/ManiSkill/scripts/verify_install_cpu.sh
~~~

该脚本固定为 num_envs=1、obs_mode=state、control_mode=pd_joint_delta_pos、sim_backend=physx_cpu、render_backend="none"、render_mode=None。它不会创建 GUI、图像观测、视频或 GPU PhysX 仿真，但会继承调用者的 CUDA_VISIBLE_DEVICES。当前源码虽在文档中称 render_backend=None 可用，但实现会在解析 None 前执行字符串操作；因此此版本实际应使用字符串 "none"。

2026-07-24 14:08 CST 首次确认 SwinAttUNet 训练 PID 693206 已结束，GPU 无 compute process。已安装 vulkan-tools；此版本的 vulkaninfo 不支持 --summary，因此改用无参数模式。保留或移除 LD_LIBRARY_PATH 的两次运行都以状态 0 成功，并枚举到 NVIDIA GeForce RTX 4090。LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 不是当前段错误的触发条件。

最小无参 RenderMaterial() 在正常环境与移除 LD_LIBRARY_PATH 的环境中均构造成功。CUDA_VISIBLE_DEVICES="" 会使 RenderMaterial(base_color=[1, 0, 0, 1]) 在材质构造时以退出码 139 段错误；相同测试在 CUDA_VISIBLE_DEVICES 未设置和 CUDA_VISIBLE_DEVICES=0 时均成功。这确认空字符串是当前 SAPIEN 3.0.3 带颜色材质崩溃的直接触发条件。

CUDA_VISIBLE_DEVICES=0 下的正式 PickCube CPU PhysX 验证已成功创建环境、reset、完成 30 step 并正常 close，日志确认 simulation backend 为 physx_cpu、render backend 为 none。官方随机动作 CPU PhysX 验证也成功运行至第 50 步 truncated，退出码 0。GPU 仅对进程可见，以满足 SAPIEN 的材质初始化；没有启用 physx_cuda。

GPU PhysX smoke test 已于 2026-07-24 20:28--20:30 CST 成功：CUDA_VISIBLE_DEVICES=0、PickCube-v1、num_envs=16、state observation、sim_backend=physx_cuda、render_backend="none"、render_mode=None。环境创建、reset、100 step 和 close 均成功；observation 为 `(16, 42)` 的 `torch.float32 cuda:0` Tensor，reward、terminated、truncated 均为 `(16,)` 批量。显存从 15 MiB 到峰值 1599 MiB，结束恢复 15 MiB，最大 GPU 利用率 67%。运行后无 GPU compute process（仅 Xorg 图形进程）。SAPIEN 仍有已知 Vulkan ICD 警告，但没有 Vulkan fatal、CUDA OOM 或 SIGSEGV，也没有 GUI、图片或视频。

## Stage 04 PPO smoke test

2026-07-24 22:47 CST 已完成一次短时 PPO 连通性验证。官方 `examples/baselines/ppo/ppo.py` 仅增加了向后兼容的 `--render-mode`、`--render-backend` 与 `--sim-backend` 参数；默认仍为原先的 `rgb_array`、`gpu`、`physx_cuda`。本次显式使用 `render_mode=None`、`render_backend="none"`、`sim_backend=physx_cuda`、state observation、PickCube-v1、num_envs=16、num_steps=64、total_timesteps=65536、4 minibatches 和 4 update epochs，禁用 WandB 与视频。

测试与外部 SwinAttUNet evaluation PID 746284 并行。运行前可用显存 18881 MiB；PPO 完成 64 个 rollout/PPO update 迭代，最终 SPS 535，policy/value/entropy 均正常且所有记录标量有限。生成了 TensorBoard event 和一个可读的 final checkpoint。总 GPU 峰值 8893 MiB、PPO 进程峰值 3768 MiB、外部进程为 5092 MiB；因此总显存和速度都不能作为 PPO 独占 benchmark。无 CUDA OOM、SIGSEGV、Vulkan fatal 或 NaN。该 smoke 只证明训练链路连通，不代表策略已经收敛，更不代表已开始正式 PPO 训练。详细记录见 `deployment_logs/PPO_SMOKE_STAGE04_SUMMARY.md`。

没有修改 NVIDIA 驱动、系统 CUDA、cuDNN、SAPIEN、PyTorch 或 Conda 环境。仅运行了上述受控 PPO smoke test，未运行正式 PPO 训练。

官方随机动作的当前源码入口与已验证命令：

~~~bash
CUDA_VISIBLE_DEVICES=0 \
MS_ASSET_DIR=/home/ubuntu/wk/maniskill_data \
MS_SKIP_ASSET_DOWNLOAD_PROMPT=1 \
/home/ubuntu/anaconda/envs/maniskill/bin/python \
  -m mani_skill.examples.demo_random_action \
  -e PickCube-v1 -o state -b physx_cpu -rb none \
  --render-mode none -n 1 -c pd_joint_delta_pos -s 0
~~~

## 当前训练隔离

受保护训练进程 PID 693206 已在 2026-07-24 14:08 CST 前结束。部署期间没有向它发送信号、没有改动其 Conda 环境，也没有修改 /home/ubuntu/wk/projects/SwinAttUNet_stage6b。

检查训练是否结束：

~~~bash
ps -p 693206 -o pid,etime,state,%cpu,%mem,rss,cmd
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits
~~~

训练运行时，禁止启动 num_envs 大于 1 的 ManiSkill 仿真、physx_cuda、PPO、SAC、RGB、RGB-D、point cloud、GUI 或 benchmark。

## 训练结束后的 GPU smoke test

仅在确认所有 GPU compute process 都已退出后，手动运行：

~~~bash
/home/ubuntu/wk/projects/ManiSkill/scripts/run_pickcube_gpu_after_training.sh
~~~

脚本会先检查 nvidia-smi 的 compute process；任一存在即拒绝执行。通过检查后，它会设置 CUDA_VISIBLE_DEVICES=0 并调用 `local_examples/verify_pickcube_gpu.py`。该验证固定为 num_envs=16、state observation、sim_backend=physx_cuda、render_backend="none"、render_mode=None，最多 100 step，且使用 PIPESTATUS 返回 Python 状态。2026-07-24 已通过一次真实服务器验证；完整日志和资源摘要在 `deployment_logs/verify_pickcube_gpu.log`、`deployment_logs/gpu_smoke_memory.log` 与 `deployment_logs/GPU_SMOKE_SUMMARY.md`。

## VS Code

VS Code 解释器配置在 .vscode/settings.json，指向独立 maniskill Python。任务位于 .vscode/tasks.json：

- ManiSkill: Environment Info
- ManiSkill: Verify PickCube CPU
- ManiSkill: Pip Check
- ManiSkill: Show Disk and GPU
- ManiSkill: GPU Smoke Test After Training
- ManiSkill: Verify PickCube GPU
- ManiSkill: Show GPU Smoke Summary
- ManiSkill: PPO Smoke After Confirmation

所有任务使用绝对路径，不依赖终端中当前激活的 Conda 环境。`PPO Smoke After Confirmation` 只可手动触发；其 Stage 04 脚本会创建时间戳子目录而不覆盖旧结果，允许外部 compute 进程存在但要求至少 10 GiB 可用显存。本次已成功运行一次：16 个训练环境、64-step rollout、65536 timesteps、state observation、`render_mode=None`、`render_backend="none"`、`sim_backend=physx_cuda`、无 WandB 上传、无视频。官方入口的无渲染 CLI 适配保持默认图像渲染行为不变。

## 上肢强化学习工作区

upper_limb_rl 目录为后续开发预留，且不包含训练产物：

~~~text
upper_limb_rl/
├── configs/    # 环境、观测、控制和训练配置
├── tasks/      # 上肢任务定义或适配层
├── training/   # PPO 启动封装和实验管理代码
└── tests/      # 无训练的环境和接口测试
~~~

建议先让 CPU PickCube 验证恢复成功，再以 state observation、小 num_envs、无视频方式实现上肢任务接口；待当前训练结束且 GPU smoke test 成功后，再评估 PPO 的设备与并行规模。官方 PPO 示例已检查，只补充了其直接导入的 tensorboard，未启动任何训练。

## 磁盘空间管理

部署前根分区约有 43G 可用空间；安装后仍应始终保留至少 20G。检查方法：

~~~bash
df -h /
du -sh /home/ubuntu/wk/projects/ManiSkill \
  /home/ubuntu/anaconda/envs/maniskill \
  /home/ubuntu/wk/maniskill_data
~~~

pip 安装使用 --no-cache-dir。不要自动清理其他项目、训练产物、/dev/shm 或系统目录。

## 完全卸载（仅供将来手动执行）

本次没有执行卸载。确认不再需要该环境和外部资产后，可由维护者手动运行：

~~~bash
/home/ubuntu/anaconda/bin/conda env remove -n maniskill
rm -rf /home/ubuntu/wk/maniskill_data
rm -rf /home/ubuntu/wk/maniskill_runs
rm -rf /home/ubuntu/wk/maniskill_cache
rm -rf /home/ubuntu/wk/maniskill_tmp
~~~

执行前务必重新确认目标路径；这些命令会删除数据。

## 常见问题

- CUDA OOM：训练运行时不要启动 GPU ManiSkill；训练结束后先用项目的 GPU smoke test，再逐步增加 num_envs。
- Vulkan：vulkan-tools 已于 2026-07-24 安装，vulkaninfo 在保留和移除 LD_LIBRARY_PATH 的条件下都可枚举 RTX 4090。当前阻塞点是 SAPIEN 3.0.3 的 RenderMaterial(base_color=...) 在 CUDA_VISIBLE_DEVICES="" 环境下段错误；后续驱动或 SAPIEN 维护须先获得明确授权。
- SAPIEN import：使用 /home/ubuntu/anaconda/envs/maniskill/bin/python，并运行 python -m pip check；不要在 swinattunet 环境安装 SAPIEN。
- PyTorch CUDA wheel：此环境固定为 torch 2.5.1+cu121；不要在该环境以外对 torch 使用 pip install -U。
- 缺少资产：检查 MS_ASSET_DIR，并确认 MS_SKIP_ASSET_DOWNLOAD_PROMPT=1；本次没有自动下载资产。
- pip 依赖冲突：运行 /home/ubuntu/anaconda/envs/maniskill/bin/python -m pip check，并保留 deployment_logs 中的完整安装日志。
- 磁盘不足：若 df -h / 显示可用空间低于 20G，停止安装或下载，先由维护者决定清理策略。
