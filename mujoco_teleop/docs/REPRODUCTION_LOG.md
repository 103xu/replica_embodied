# 复现记录：2026-10-29

## 目标

在 `C:\Users\32010\GitHub\replica_embodied` 复现 Every-Embodied 的
`04mujoco复现ACT、Pi0、SmolVLA` 第 1、2 部分：OMY 键盘示教采集与数据回放。
保留运行所需代码、模型和资源，并说明实际做过的验证及限制。

## 1. 工作区与环境调查

- 初始 Git 工作区干净，已有 `mujoco_demo/` 与根 README。
- 创建分支 `codex/omy-collection-replay`，原项目不改动。
- base 为 Python 3.14.6，与教程固定旧依赖不匹配。
- 已有 my_env 为 Python 3.10.21，可直接通过解释器绝对路径调用。
- `conda.exe` 调用遇到 Windows DLL 重定位错误，因此使用现有解释器，不修复或重装 Anaconda。
- 检测到 RTX 5060 Laptop GPU，`nvidia-smi` 报告显存 8151 MiB。
- 原有 torch `2.14.0+cu130`、torchvision `0.29.0+cu130`，保留原版本。
- 原有环境包清单记录为 `environment-before.json`。

## 2. 固定上游版本与获取文件

- Every-Embodied 固定提交：`14d7c6c93debf01419b21e50b649a06f15881c38`。
- 读取第 1、2 部分 Notebook，确认采集观测为末端位姿，动作标签来自
  `get_commanded_joint_action()`，不能使用 `step()` 的返回值。
- 按清单下载并验证 72 个原始文件，约 37.7 MB（含盘子 ZIP）。
- 保留 OMY 模型、杯子模型、桌面/地面资源、环境/IK/变换代码和两个原始 Notebook。
- 解压 `plate_11.zip`，过滤 macOS 元数据并校验解压目标不越界。
- 保留上游 LICENSE 和第三方来源说明。

## 3. 本地适配

- 使用 my_env 解释器；新增依赖安装到项目 `.deps`，不替换现有环境包。
- 保持 MuJoCo 3.1.6 与 LeRobot 固定提交；不安装教程中的旧版 PyTorch/CUDA。
- 第一次依赖解析发现 PyAV 14.4.0 没有 cp310 Windows 轮子，改为有预编译包的 14.2.0，并要求二进制轮子以避免本机编译。
- LeRobot 只安装本章实际使用的数据组件依赖，不安装训练、真机和在线日志等无关组件。
- 提供独立 `collect`、`replay`、`smoke`、`doctor` 命令。
- 用单进程加两个图像写入线程替代上游多进程写图，回放不创建 DataLoader 子进程。
- 采集路径已存在即拒绝覆盖，不保留原 Notebook 中的递归删除行为。
- 固定每帧推进 0.05 秒物理仿真，保持数据的 20 Hz 语义。
- 观测在准备当前动作前采集，标签存储实际下发的目标关节角和经过速率限制的夹爪命令。
- 额外保存初始 MuJoCo 积分状态，供本地确定性回放使用。
- 保存数据到标准 LeRobot v2.1 Parquet/JSONL 结构，运行时默认关闭网络访问。

## 4. 验证结果

### 4.1 环境与窗口

- `run.ps1 doctor` 成功，确认运行解释器来自 my_env、MuJoCo 3.1.6，CUDA 可用。
- 有窗口 smoke 模式成功保存 12 帧，验证主窗口、相机叠层与正常关闭。
- `collect --max-seconds 2` 打开正式采集窗口后正常退出，没有操作、没有成功回合，保存数量为 0；不把这次启动检查算作人类成功示教。
- `docs/environment-preservation.json` 证明复现前后 my_env 已安装包清单一致，base 未改动。

### 4.2 接口 smoke

- 保存并回放 `data/smoke_001` 的 40 帧测试数据。
- `complete=true`，位置对齐最大误差约 `1.19357e-7 m`。
- 任务成功为 false，符合该数据仅测试上下小幅运动的预期。
- 原始结果保存在 `docs/smoke_replay_report.json`，不是正式抓放示教。

### 4.3 完整抓放、发现的问题与修复

- 先用脚本 IK 实际抓杯：接近、下降、闭爪、抬升、搬运、下降、松爪、抬臂。没有在运行期间修改杯子位置，也没有增加吸附约束。
- 前两次抓取高度过高，杯子被推开而没有抬升；调整 TCP 与指垫之间的高度偏差后，实际抓取成功。
- 初版成功示教为 628 帧。回放虽然满足上游成功判定，但最大末端位置偏差约 1.10 cm，杯子最终姿态也有差异，因此未作为最终结果。
- 把放下高度从 0.87 m 调为 0.91 m，减少夹持阶段将杯子压入桌面/盘子的情况。
- 检查发现上游 IK 仅恢复 qpos/qvel/act，`mj_forward` 会改变 22 个求解器热启动分量；单步检查中最大变化约 1.95。此副作用并未记录在数据动作中。
- 本地 `OMYEnvironment.step()` 现在恢复完整 `mjSTATE_INTEGRATION` 状态，刷新派生变换后再次恢复热启动数据，保持命令准备不推进或改变物理状态。
- 进一步将实际执行命令量化为数据中同样的 float32，并从该命令构造关节控制，避免 float64 IK 目标与 float32 标签在接触阶段产生不同轨迹。
- 上游 vendor 文件保持原样，修复集中在 `src/omy_repro/runtime.py`。

### 4.4 最终结果

| 项目 | 实际结果 |
| --- | --- |
| 数据目录 | `data/scripted_demo_004` |
| 控制方式 | 脚本 IK，非 VLA 策略，非人类键盘录制 |
| 种子 / 采集频率 | 0 / 20 Hz |
| 保存回合 / 帧数 | 1 / 602 |
| 对应仿真时长 | 30.1 秒 |
| 杯子最大抬升 | 0.2068942576 m |
| 采集成功判定 | true |
| 完整回放 | 602 / 602 帧，complete=true |
| 回放成功判定 | true |
| 最大下一帧末端位置误差 | 0.0 m（同机实测） |
| 回归检查 | 5 项全部通过 |

结构化证据：`docs/scripted_collection_report.json`、`docs/scripted_replay_report.json`、
`docs/environment-preservation.json`。末帧截图保存在 `docs/media/`。
完整采集动画保存在本机 `outputs/scripted_preview.gif`。

回归检查覆盖：旧数据不覆盖；标签是实际执行目标且与后续命令不共享内存；完整物理状态恢复；IK 准备动作不改变积分状态；Parquet 图像、动作、时间戳往返一致。

## 5. 当前保留的文件和范围

- 源码、固定依赖、原始 Notebook、模型、来源清单、README、验证报告都保留。
- `.deps/`、`.downloads/`、`data/`、`outputs/` 通过 `.gitignore` 排除，当前磁盘上仍然保留。
- `data/scripted_demo_001` 至 `003` 是调试版本，正式演示命令使用 `004`；不用旧数据评估最终修复。
- 没有完成手工键盘抓放录制；键盘入口已验证启动，用户可按 README 自己采集。
- 本次不训练 ACT / SmolVLA，不需要重新安装 Ubuntu、VMware 或 WSL。
- 成功沿用教程的位置/夹爪/末端高度规则，不代表对杯子姿态、稳定性或泛化能力作严格评估。
- 本机回放对齐不能直接推广为跨系统、显卡、MuJoCo 版本逐位一致。
- 初次复现交付时未执行 Git 提交或远端推送。根目录 README 已增加本项目入口；用户随后授权上传到 `103xu/replica_embodied`，发布范围为源码、必要模型、README 与验证记录，本地依赖、下载缓存和示教数据继续忽略。
- 盘子原始网格较大，Git 保留上游 ZIP；解压出的 `plate_11/` 留在本机并忽略，启动时支持离线自动解压。避免把大型 OBJ 直接提交到 GitHub。
- 最后以项目内临时目录模拟全新克隆，离线解压盘子资源通过；72 项上游资源哈希复核通过，Python 编译检查和 `git diff --check` 通过。Git 将保留的最大单文件为上游 ZIP（13,119,540 字节）。
