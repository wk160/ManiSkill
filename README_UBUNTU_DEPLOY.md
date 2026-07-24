# ManiSkill 3 Ubuntu 隔离部署记录

部署日期：2026-07-22  
主机：ubuntu-server  
仓库：/home/ubuntu/wk/projects/ManiSkill  
Git 分支：feature/upper-limb-rl  
Git HEAD：42b6824 [BugFix] fix scalar-seeded main RNG expansion shrinking the episode seed batch (#1457)

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

CUDA_VISIBLE_DEVICES=0 下的正式 PickCube CPU PhysX 验证已成功创建环境、reset、完成 30 step 并正常 close，日志确认 simulation backend 为 physx_cpu、render backend 为 none。官方随机动作 CPU PhysX 验证也成功运行至第 50 步 truncated，退出码 0。GPU 仅对进程可见，以满足 SAPIEN 的材质初始化；没有启用 physx_cuda。GPU smoke test 本次未运行。没有修改 NVIDIA 驱动、系统 CUDA、cuDNN、SAPIEN、PyTorch 或 Conda 环境，也没有运行 PPO。

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

脚本会先检查 nvidia-smi 的 compute process 和 PID 693206；任一存在即拒绝执行。通过检查后它才会以 num_envs=16、state observation、无 GUI、短时 step 的方式执行 PhysX CUDA smoke test。该脚本本次没有运行。

## VS Code

VS Code 解释器配置在 .vscode/settings.json，指向独立 maniskill Python。任务位于 .vscode/tasks.json：

- ManiSkill: Environment Info
- ManiSkill: Verify PickCube CPU
- ManiSkill: Pip Check
- ManiSkill: Show Disk and GPU
- ManiSkill: GPU Smoke Test After Training

所有任务使用绝对路径，不依赖终端中当前激活的 Conda 环境。最后一项只可手动触发，且只应在训练结束后使用。

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
