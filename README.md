# YOLO Gray1 训练环境

本仓库提供已在 NVIDIA H200 上验证的单通道姿态模型标准训练环境。模型为 grayscale-native YOLOv8x-pose-P6，固定输入 ABI 为 `[B, 1, 1280, 1280]`，包含 21 类目标，每个目标 2 个关键点；模型从单通道结构初始化，不加载三通道预训练权重。仓库、数据和运行产物可以放在任意用户有权限的绝对路径下。

正式镜像：

```text
ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

本仓库保存训练代码、容器定义和小型配置文件。Docker 镜像归档、离线运行时与模型权重等大文件保存在 NAS，不进入 GitHub。

## 端到端复现

请按编号顺序阅读和执行：

1. [准备并校验数据](docs/human/01_DATA_PREPARATION.md)
2. [启动并验证训练](docs/human/02_RUN_TRAINING.md)
3. [理解训练参数与配置](docs/human/03_TRAINING_CONFIGURATION.md)
4. [从 GitHub 和 NAS 恢复环境](docs/human/04_RESTORE_ENVIRONMENT.md)

烟测只验证环境、数据、训练、checkpoint 保存和 final evaluation 能否贯通，不用于验证模型精度。

自动化 Agent 请从 [AGENT.md](AGENT.md) 开始，不要以本文替代操作约束。
