# 训练参数与配置

本文是端到端复现的第 3 步，用于理解和审计训练参数。第一次复现请先完成[第 2 步：启动并验证训练](02_RUN_TRAINING.md)。

## 1. 参数覆盖顺序

训练参数按以下顺序逐级覆盖，越靠后优先级越高：

```text
Ultralytics 默认配置
→ 模型结构与数据集配置
→ 项目训练脚本的默认值和固定值
→ 启动命令行参数
→ 本次运行最终记录 args.yaml
```

排查某次训练时，先查看该运行的 `args.yaml`，再反查启动命令和源码。

## 2. 命令行参数

| 参数 | 入口默认值 | 标准烟测值 | 含义与约束 |
|---|---:|---:|---|
| `--data` | 必填 | `/config/dataset.yaml` | 数据集配置；必须满足 Gray1 契约 |
| `--model-config` | Gray1 P6 配置 | 默认值 | 模型结构文件；必须为 `ch: 1`、`nc: 21`、`kpt_shape: [2, 3]` |
| `--epochs` | `150` | `1` | 完整遍历训练集的次数；烟测只跑 1 次 |
| `--imgsz` | `1280` | `1280` | 输入边长；Gray1 ABI 固定为 1280，其他值会被拒绝 |
| `--batch` | `4` | `4` | 每批图像数；`-1` 表示 Ultralytics AutoBatch |
| `--device` | `cpu` | `0` | 计算设备；容器仅暴露一张 GPU，因此其容器内编号为 0 |
| `--workers` | `4` | `4` | DataLoader 子进程数；0 表示主进程加载 |
| `--project` | `/experiments` | `/experiments` | 容器内输出根目录 |
| `--name` | 固定默认名称 | `$RUN_ID` | 本次实验子目录名称 |
| `--save-period` | `10` | `1` | 每隔多少 epoch 保存周期 checkpoint |
| `--resume` | 未设置 | 未设置 | 仅允许兼容的 Gray1 checkpoint |

查看入口支持的参数不会使用 GPU：

```bash
docker run --rm \
  --network none \
  --read-only \
  --entrypoint python \
  yolo-gray1:1280-train-predict-20261016 \
  /workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py \
  --help
```

## 3. 项目固定训练行为

训练入口固定启用：

- `amp=True`：自动混合精度。
- `patience=100`：早停等待周期。
- `save=True`、`plots=True`：保存权重和图表。
- `cache=False`：关闭缓存。
- `multi_scale=True`：默认在 `0.5x` 到 `1.5x` 基准 `imgsz` 之间进行动态多尺度训练；可用 `--no-multi-scale` 关闭。

默认开启在线空间增强：`mosaic=1.0`、`mixup=0.1`、`flipud=0.5`、`fliplr=0.5`、旋转 `10°`、平移 `0.1`、缩放 `0.5`、错切 `2°`、透视 `0.0005`，以及 Pose 数据管线中的随机擦除 `0.2`。灰度输入的 HSV 与 BGR 交换保持关闭。`copy_paste=0.1` 会被记录，但原生 Ultralytics Copy-Paste 需要实例分割轮廓，当前 Pose 标签没有轮廓，因此不会修改图像。

验证像素误差统一为每个关键点的纯 L1：`|x_pred-x_gt| + |y_pred-y_gt|`。训练中的坐标辅助损失仍为 Smooth L1，两者不要求数值相等。

当前使用 `optimizer=auto`，因此 Ultralytics 可能自动选择优化器、学习率和 momentum，并忽略默认 `lr0`/`momentum`。最终实际值以日志和 `args.yaml` 为准。

## 4. 关键配置文件

| 宿主机路径 | 容器内路径 | 内容 |
|---|---|---|
| `src/Yolo_fine/train_yolov8x_p6_pose_gray1.py` | `/workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py` | 参数解析、契约校验、固定训练行为 |
| `src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml` | `/workspace/src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml` | 单通道模型结构 |
| `$DATASET/dataset.yaml` | 复制并映射为 `/config/dataset.yaml` | 类别、数据划分和 Gray1 数据契约 |
| `src/Yolo_fine/ultralytics/ultralytics/cfg/default.yaml` | 同一 `/workspace` 相对路径 | Ultralytics 未被覆盖的默认值 |
| `$RUN_ROOT/experiments/$RUN_ID/args.yaml` | 运行时生成 | 本次训练最终生效参数 |
| `docker/requirements.lock.txt` | 构建输入 | Python 依赖锁定清单 |
| `docs/agent/runtime-packages-ultralytics-8.3.98-v1.json` | 发布记录 | 关键运行时版本 |

模型和数据集必须共同保持：

```yaml
ch: 1                 # 模型配置
nc: 21                # 模型配置
kpt_shape: [2, 3]     # 模型与数据集配置
channels: 1           # 数据集配置
input_semantics: gray1
```

查看关键配置：

```bash
REPO_ROOT='<absolute-path-to-yolo-gray1-train>'
cd "$REPO_ROOT"
sed -n '1,240p' src/Yolo_fine/train_yolov8x_p6_pose_gray1.py
sed -n '1,260p' src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml
sed -n '1,320p' src/Yolo_fine/ultralytics/ultralytics/cfg/default.yaml
```

## 5. 调整参数时的边界

烟测脚本固定为单 GPU、1 epoch、batch 4 和 55 分钟硬超时。生产规模训练若增加 epoch、batch、worker 或运行时长，应重新评估共享 GPU、主机内存、`/dev/shm`、数据读取吞吐和输出空间。不得通过修改 `imgsz`、通道数、类别数或关键点形状绕过 Gray1 ABI。

返回：[项目首页](../../README.md) · 上一步：[启动并验证训练](02_RUN_TRAINING.md) · 下一步：[恢复环境](04_RESTORE_ENVIRONMENT.md)
