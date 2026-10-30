# 复刻仿真项目
作为初学者，这个仓库用来装我正在学习的具身智能项目

最新项目：[具身智能与 VLA 仿真实验阶段汇报](mujoco_teleop/docs/VLA_TEACHER_OVERVIEW.md)

自学入口：[从机械臂示教到理解 VLA 的个人练习册](mujoco_teleop/docs/LEARNER_GUIDE.md)（按教程顺序，包含可执行命令、预期效果和原理解释）。

## 现在已有项目
- [x] 初识 mujoco 的 demo → [`mujoco_demo/`](mujoco_demo/)（UR 机械臂 + 夹爪抓取演示，安装与运行说明见 [mujoco_demo/README.md](mujoco_demo/README.md)）
- [x] OMY 示教采集与回放 → [`mujoco_teleop/`](mujoco_teleop/)(以下均在此目录下)
- [x] ACT 训练与仿真部署 → 保留 10 步配置的失败对照及 100 步 ACT 抓放成功配置。运行说明见 [mujoco_teleop/README.md](mujoco_teleop/README.md)，记录见 [ACT_REPRODUCTION_LOG.md](mujoco_teleop/docs/ACT_REPRODUCTION_LOG.md)。
- [x] 语言条件数据与回放 → 20 条红/蓝杯脚本示教及物理审计。
- [x] SmolVLA 训练与部署链路 → 教程公开 OMY 权重在接口与相机对齐后，红/蓝任务物理成功 2/2；自己的两套微调模型仍未成功，分别记录于 [VLA_REPRODUCTION_LOG.md](mujoco_teleop/docs/VLA_REPRODUCTION_LOG.md)。
- [x] π0.5 数据准备 → 独立 Python 3.12 / LeRobot 0.6.1 CPU 环境，v2.1→v3 转换、动作窗口对齐及 6 项接口测试。
- [ ] π0、π0.5 完整微调与部署 → 本机约 8GB 显存，且 π0 依赖的 PaliGemma 匿名配置访问未获授权；尚未完成。
- [ ] XR 遥操作扩展 → 已保留上游源码，未配置 XR 设备及 TeleopXR。
