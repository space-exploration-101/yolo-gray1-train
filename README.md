# YOLO Gray1 训练环境

本仓库提供 H200 上单通道姿态模型的标准训练环境。模型为 grayscale-native YOLOv8x-pose-P6，输入固定为 `[B, 1, 1280, 1280]`，包含 21 类目标和每个目标 2 个关键点。模型默认从单通道结构初始化，不加载三通道预训练权重。

正式镜像：

```text
ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

## 关键文件

| 路径 | 用途 |
|---|---|
| `run-gray1-training-smoke.sh` | 单卡、1 epoch、55 分钟硬超时的标准烟测入口 |
| `src/Yolo_fine/train_yolov8x_p6_pose_gray1.py` | 训练入口和参数约束 |
| `src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml` | 单通道模型结构 |
| `docker/Dockerfile` | 正式镜像构建文件 |
| `docker/requirements.lock.txt` | Python 依赖锁定清单 |
| `docs/runtime-packages-ultralytics-8.3.98-v1.json` | 关键运行时版本 |
| `RELEASE_SHA256SUMS` | 完整发布文件校验和 |

## 训练数据要求

训练只能使用已完成预处理和严格验证的 prepared dataset：

```text
<prepared-dataset>/
├── VERIFIED
├── dataset.yaml
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

`VERIFIED` 必须由验证流程生成，不能只手工创建空文件。`dataset.yaml` 必须满足：

```yaml
channels: 1
input_semantics: gray1
kpt_shape: [2, 3]
```

同时必须包含恰好 21 个类别名称，不得启用旧的 `r_channel_only` 模式。图像应能按单通道灰度图读取，图像和 YOLO pose 标签必须同名配对。

训练前检查：

```bash
DATASET='<prepared-dataset>'
test -f "$DATASET/VERIFIED"
test -f "$DATASET/dataset.yaml"
test -d "$DATASET/images/train"
test -d "$DATASET/images/val"
test -d "$DATASET/labels/train"
test -d "$DATASET/labels/val"
grep -E '^(channels|input_semantics):' "$DATASET/dataset.yaml"
```

## 在 H200 上运行

登录并进入目录：

```bash
ssh H200
cd /data3/ywang/yolo-gray1-train
docker image inspect ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

H200 是共享服务器。使用 GPU 前必须检查利用率、显存、活动进程及 owner，不能依据单次 `0%` 判断空闲：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi dmon -s pucm -c 5
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

启动标准烟测：

```bash
GPU_UUID='<confirmed-idle-gpu-uuid>'
DATASET='<prepared-dataset>'
RUN_ID="gray1_smoke_$(date +%Y%m%d_%H%M%S)"

./run-gray1-training-smoke.sh "$GPU_UUID" "$DATASET" "$RUN_ID"
```

脚本只向容器暴露所选的一张 GPU；它在容器内显示为 `device 0`。脚本还会检查目标 GPU 利用率不高于 20%、空闲显存不少于 32768 MiB且没有活动计算进程。不要抢占或终止其他用户任务。

成功时会打印：

```text
TRAINING_SMOKE_PASS model=...
```

模型位于：

```text
/data1/ywang/workflow-test/runs/<RUN_ID>/experiments/<RUN_ID>/weights/best.pt
```

该烟测只验证环境、数据、训练、checkpoint 保存和 final evaluation 能否贯通，不用于验证模型精度。

## 关键参数

| 参数 | 烟测值 | 说明 |
|---|---:|---|
| `--epochs` | `1` | 训练入口默认 150；烟测只跑 1 epoch |
| `--imgsz` | `1280` | Gray1 ABI 固定值，其他值会被拒绝 |
| `--batch` | `4` | 每个 batch 的图像数 |
| `--device` | `0` | 容器内唯一可见 GPU |
| `--workers` | `4` | DataLoader worker 数量 |
| `--save-period` | `1` | 烟测每个 epoch 保存 checkpoint |
| `--resume` | 未设置 | 仅允许兼容的 Gray1 checkpoint |

入口固定启用 AMP，关闭 cache、multi-scale 及 HSV、翻转、mosaic、mixup 等在线增强，因为数据应已在离线预处理阶段完成增强。完整最终参数保存在每次运行目录的 `args.yaml`。生产规模训练如需增加 epoch、调整 batch 或延长超时，必须重新评估共享 GPU 占用。

## 大文件备份与恢复

GitHub 只保存源码和小文件。以下大文件保存在 NAS：

```text
Windows:
\\10.2.26.26\902_data\0-项目\13-专项\4-代码\训练平台\yolo-gray1-train

H200:
/mnt/ywang-nas/0-项目/13-专项/4-代码/训练平台/yolo-gray1-train
```

NAS 中保存正式 Docker 镜像归档、离线运行时和未纳入 Git 的大权重。恢复前先校验 NAS：

```bash
NAS_ROOT='/mnt/ywang-nas/0-项目/13-专项/4-代码/训练平台/yolo-gray1-train'
cd "$NAS_ROOT"
sha256sum -c SHA256SUMS
zstd -q -t docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst
```

恢复镜像：

```bash
zstd -dc "$NAS_ROOT/docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst" | docker load
docker image inspect ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

如需从 Dockerfile 重建，先把 NAS 中的文件按原相对路径复制回仓库，再执行：

```bash
cd /data3/ywang/yolo-gray1-train
rsync -a "$NAS_ROOT/docker/yolo-runtime.tar.gz" docker/
rsync -a "$NAS_ROOT/src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt" src/Yolo_fine/ultralytics/
docker build -f docker/Dockerfile -t ywang/yolo-gray1-train:ultralytics-8.3.98-v1 .
```

发布目录完整时可执行：

```bash
cd /data3/ywang/yolo-gray1-train
sha256sum -c RELEASE_SHA256SUMS
```
