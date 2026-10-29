# π0.5 OMY 远程五步微调检查

2026-10-04，在 的 Ubuntu 工作站完成真实预训练参数加载、五次反向更新、保存、重新加载及留出图像推理检查。运行使用本章 `pi05_omy_utils.py` 生成的命令和 `pi05_omy_train.py` 入口。

## 权重与环境

- 显卡：NVIDIA RTX PRO 6000 Blackwell Workstation Edition，96 GB 显存。
- 独立环境：`/data/Data14TB/envs/every-embodied-omy-pi05-061`；Python 3.12.3、LeRobot 0.6.1、PyTorch 2.7.1、CUDA［并行计算平台］12.8、MuJoCo 3.1.6。
- **实际起点为工作站缓存的官方 OpenPi π0.5 LIBERO 预训练策略**：`/data/Data14TB/robotics_shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors`，7,233,650,408 字节、812 个张量。
- 本次没有使用通用 `lerobot/pi05_base` 权重。LIBERO 预训练策略已针对其他任务微调，五步结果只证明该起点与当前 OMY 训练接口兼容。

实验根目录：

```text
/data/Data14TB/robotics_shared/experiments/omy-pi05-notebook-20261004/short_finetune_20261004
```

其 `pi05_libero_start` 目录使用符号链接引用原权重，配套三个 LeRobot 配置文件取自 `lerobot/pi05_base` 的 `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` 版本。缓存权重与该模型的键名、形状匹配；官方加载器映射 OpenPi 键名后严格加载，日志明确显示 `All keys loaded successfully!`。原预训练模型保留在原目录。

本章对固定版加载器增加成功检查，读取失败或参数匹配失败会停止。对应测试覆盖完整加载和官方加载器返回随机初始化模型的两种失败情形。

## 数据与训练设置

在真实 MuJoCo OMY 单杯场景中新采集三条短轨迹，每条 20 帧，共 60 帧，频率为 20 Hz［每秒 20 次控制］。关节目标在初始位形附近小幅变化，夹爪同步开合；这些轨迹没有完成抓放任务，只用于训练链路检查。

状态为六维末端位姿，动作为实际下发的六个绝对目标关节角与归一化夹爪命令，两路图像为外部、腕部相机的 256×256 图像。记录时序与修复后的第 1 节采集逻辑一致。

- 原数据：实验根目录下的 `real_omy_v21`。
- 独立转换数据：`real_omy_v3`，训练回合为 0、1，共 40 帧；留出回合为 2，共 20 帧。
- 归一化分位数只使用训练回合计算；动作块与回合末尾填充通过真实新版读取接口检查。
- 更新次数 5、批量大小 1、预测动作块长 50、连续执行步数 5。
- 关闭相对动作，仅训练动作专家、冻结视觉编码器、开启梯度检查点、使用 BF16［16 位浮点格式］；关闭模型编译、在线平台上传和训练时环境评测。

## 实际结果

| 更新步 | 损失 | 梯度范数 | 峰值分配显存（GiB［显存容量单位］） |
| --- | ---: | ---: | ---: |
| 1 | 1.332 | 11.550 | 12.85 |
| 2 | 1.446 | 6.752 | 12.85 |
| 3 | 1.100 | 9.364 | 12.85 |
| 4 | 0.979 | 3.011 | 12.85 |
| 5 | 0.922 | 3.350 | 12.85 |

显存数值来自训练器的 `torch.cuda.max_memory_allocated()`，表示该进程分配张量的峰值，不是整张显卡的总占用。五次更新的损失、梯度均为有限值，训练进程退出码为 0。

保存模型与起点逐元素比较后：动作输入层有 32,763/32,768 个元素变化，动作输出层有 32,768/32,768 个元素变化，最大变化约为 `8e-5`；抽查的冻结视觉投影层没有变化。模型确实执行了参数更新。

训练状态保存的更新步为 5。保存模型重新严格加载成功，保存的归一化、反归一化处理器通过动作往返检查；对留出回合的第一帧图像与状态执行推理，得到形状 `(1, 50, 7)` 的有限值动作。首次推理约 0.86 秒，此数值不能作为实时部署性能结论。

原模型有 812 个张量，保存模型有 813 个张量，多出框架保存时展开的共享语言嵌入项；重新严格加载无缺失项或多余项。

完整结果目录：

```text
train-5steps/checkpoints/000005/pretrained_model
train-5steps/checkpoints/000005/training_state
train-5steps/checkpoints/last -> 000005
```

训练日志、命令、测试输出和数值检查结果的本地副本见 [validation/pi05_omy_20261004](validation/pi05_omy_20261004)，远程原件位于实验根目录。六项辅助程序测试全部通过。

## 复用本地模型

在该工作站运行笔记本前，可设置以下环境变量来复用本次准备的兼容目录：

```bash
export OMY_PI05_PRETRAINED=/data/Data14TB/robotics_shared/experiments/omy-pi05-notebook-20261004/short_finetune_20261004/pi05_libero_start
export HF_HOME=/data/Data14TB/robotics_shared/cache/huggingface
```

仍需按照笔记本设置自己的示教源、转换目录和新的输出目录。原始 OpenPi 目录中的简化配置不是完整的 LeRobot 配置，不能直接替代这里准备的适配目录。

本次通过的是微调链路检查，**没有评测 OMY 抓放成功率，也没有验证完整训练效果**。完整复现需重新采集覆盖不同初始位置的成功示教，按回合留出评测，然后完成正式训练和闭环抓放测试。
