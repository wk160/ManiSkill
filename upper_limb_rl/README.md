# 上肢强化学习开发区

本目录只预留 ManiSkill 3 上肢强化学习代码与配置，不存放数据集、模型权重、视频或训练运行产物。

- configs：环境、观测、控制器和训练配置。
- tasks：上肢任务实现或 ManiSkill 适配层。
- training：PPO 启动封装、评估与实验管理。
- tests：不运行长训练的接口和环境测试。

开发前提：先恢复并通过仓库根目录 README_UBUNTU_DEPLOY.md 所述 CPU PickCube 验证；训练服务器有其他 GPU compute process 时，不得启动 GPU 并行仿真或 PPO。
