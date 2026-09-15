# 启动训练

本文是端到端复现的第 2 步，说明如何在 H200 上运行单卡 Gray1 训练烟测。开始前必须完成[第 1 步：准备并校验数据](01_DATA_PREPARATION.md)。正式镜像为：

```text
ywang/yolo-gray1-train:ultralytics-8.3.98-v1
```

## 1. 训练前检查

登录 H200，进入正式仓库并指定 prepared dataset：

```bash
ssh H200
cd /data3/ywang/yolo-gray1-train

DATASET='<prepared-dataset>'
test -f "$DATASET/VERIFIED"
test -f "$DATASET/dataset.yaml"
test -d "$DATASET/images/train"
test -d "$DATASET/images/val"
test -d "$DATASET/labels/train"
test -d "$DATASET/labels/val"
docker image inspect ywang/yolo-gray1-train:ultralytics-8.3.98-v1 >/dev/null
```

若使用随工作流发布的合成烟测集：

```bash
DATASET=/data1/ywang/workflow-test/synthetic-pose21-r-only-v1/prepared
```

H200 是共享服务器。选卡前同时检查利用率、显存、计算进程和进程所有者，不能仅凭一次 `0%` 判断空闲：

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi dmon -s pucm -c 5
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

如存在计算进程，可按 PID 查询 owner；不要停止或抢占其他用户任务：

```bash
ps -o user,pid,etime,stat,cmd -p '<gpu-process-pid>'
```

## 2. 推荐启动方式

选择一张确认空闲的 GPU，使用完整 UUID：

```bash
GPU_UUID='<confirmed-idle-gpu-uuid>'
RUN_ID="gray1_smoke_$(date +%Y%m%d_%H%M%S)"

./run-gray1-training-smoke.sh "$GPU_UUID" "$DATASET" "$RUN_ID"
```

脚本会完成以下操作：

1. 验证 GPU UUID、数据集标记、镜像和目标目录。
2. 对 GPU 连续采样，并要求利用率不高于 20%、空闲显存不少于 32768 MiB且没有活动计算进程。
3. 创建独立运行目录并生成容器使用的 `dataset.container.yaml`。
4. 只向容器暴露一张 GPU，运行 1 epoch，并设置 55 分钟硬超时。
5. 完成训练与 final evaluation，确认 `best.pt` 存在并输出 SHA-256。

第三个参数可省略；脚本会自动生成 `e2e_smoke_<timestamp>`。已有运行目录或容器名称不会被覆盖。

## 3. 输出和成功条件

运行根目录：

```text
/data1/ywang/workflow-test/runs/<RUN_ID>/
```

主要产物：

```text
dataset.container.yaml
experiments/<RUN_ID>/args.yaml
experiments/<RUN_ID>/weights/best.pt
experiments/<RUN_ID>/weights/last.pt
```

成功时脚本打印：

```text
TRAINING_SMOKE_PASS model=<absolute-path-to-best.pt>
```

手工复核：

```bash
RUN_ROOT="/data1/ywang/workflow-test/runs/$RUN_ID"
MODEL="$RUN_ROOT/experiments/$RUN_ID/weights/best.pt"
test -s "$MODEL"
sha256sum "$MODEL"
cat "$RUN_ROOT/experiments/$RUN_ID/args.yaml"
```

`final evaluation` 会重新加载训练后保存的最佳权重，执行最终验证并确认模型可用于推理/验证链路。它是烟测成功条件的一部分，不代表模型精度达到生产要求。

## 4. 手工容器命令

通常应使用脚本。需要审计或排错时，可以查看它生成的完整命令：

```bash
sed -n '1,240p' /data3/ywang/yolo-gray1-train/run-gray1-training-smoke.sh
```

其中关键隔离措施包括：数据集只读挂载、实验目录可写、容器根文件系统只读、无网络、单 GPU UUID 绑定以及 55 分钟硬超时。容器只看见所选的一张卡，所以训练参数使用 `--device 0`。

## 5. 常见问题

- `dataset is not verified`：预处理校验没有成功完成，参见[准备并校验数据](01_DATA_PREPARATION.md)。
- `GPU gate failed` 或检测到计算进程：换用确认空闲的 GPU，不能终止他人进程。
- `run directory` 或容器名称已存在：生成新的 `RUN_ID`，不要复用旧目录。
- 训练中断：保留运行目录和日志；重新烟测应使用新 `RUN_ID`。
- TensorBoard graph 的 3 通道 warmup 警告不应再导致 final evaluation 失败；若再次出现通道异常，应保留完整日志并停止扩大训练规模。

返回：[项目首页](../../README.md) · 下一步：[理解训练参数与配置](03_TRAINING_CONFIGURATION.md)
