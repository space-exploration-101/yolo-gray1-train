# YOLO Gray1 训练环境

本仓库提供在 NVIDIA H200 上运行的单通道姿态模型标准训练环境。模型为 grayscale-native YOLOv8x-pose-P6，固定输入 ABI 为 `[B, 1, 1280, 1280]`，包含 21 类目标，每个目标 2 个关键点；模型从单通道结构初始化，不加载三通道预训练权重。

正式镜像：

```text
ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

本仓库保存训练代码、容器定义和小型配置文件。Docker 镜像归档、离线运行时与模型权重等大文件保存在 NAS，不进入 GitHub。

## 使用文档

- [数据预处理](docs/DATA_PREPARATION.md)：原始数据要求、烟测数据选择、构建和严格校验。
- [启动训练](docs/TRAINING.md)：训练前检查、GPU 门禁、标准脚本和产物确认。
- [训练参数与配置](docs/TRAINING_PARAMETERS.md)：参数含义、覆盖顺序和关键配置文件。
- [环境恢复](docs/ENVIRONMENT_RECOVERY.md)：GitHub 与 NAS 的备份位置、校验、镜像恢复和重建。

烟测只验证环境、数据、训练、checkpoint 保存和 final evaluation 能否贯通，不用于验证模型精度。
