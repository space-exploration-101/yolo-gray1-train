# 训练入口、参数和验收位置

## 直接回答“如何运行”

人类用户优先使用：

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
RUNS_ROOT='<absolute-writable-runs-directory>'
cd "$REPO_ROOT"
TRAIN_RUNS_ROOT="$RUNS_ROOT" \
./run-gray1-training-smoke.sh \
  '<confirmed-idle-gpu-uuid>' \
  '<prepared-dataset-path>' \
  "gray1_smoke_$(date +%Y%m%d_%H%M%S)"
```

完整前置门禁和解释位于 `docs/human/02_RUN_TRAINING.md`。不要仅给出训练命令而省略共享 GPU 检查。

## 参数权威顺序

```text
Ultralytics default.yaml
→ 模型 YAML 与 dataset.yaml
→ train_yolov8x_p6_pose_gray1.py 默认值/固定值
→ run-gray1-training-smoke.sh 命令行值
→ 运行产物 args.yaml
```

常见问题的最快定位：

| 问题 | 位置 |
|---|---|
| 脚本实际传入什么参数 | `run-gray1-training-smoke.sh` 中 `docker run` 尾部 |
| 参数默认值和允许范围 | `train_yolov8x_p6_pose_gray1.py::parse_args`、`validate_args` |
| 固定增强、AMP、cache 等 | `train_yolov8x_p6_pose_gray1.py::main` 的 `train_kwargs` |
| 模型结构 | `configs/yolov8x-pose-p6-gray1.yaml` |
| 未覆盖的 Ultralytics 默认值 | `ultralytics/ultralytics/cfg/default.yaml` |
| 某次运行最终值 | `$RUNS_ROOT/<RUN_ID>/experiments/<RUN_ID>/args.yaml` |

标准烟测固定为 `epochs=1`、`imgsz=1280`、`batch=4`、`device=0`、`workers=4`、`save_period=1`，外层硬超时 55 分钟。容器内 `device=0` 指宿主机按 UUID 显式绑定的唯一 GPU。

`TRAIN_RUNS_ROOT` 是可配置的绝对输出根；未设置时默认为调用脚本时 `$PWD/runs`。容器名使用 `gray1-train-<RUN_ID>`，不绑定宿主机用户名。

## 成功与失败判据

成功必须同时满足：

- 训练进程退出码为 0。
- final evaluation 完成。
- `weights/best.pt` 存在且非空。
- 封装脚本输出 `TRAINING_SMOKE_PASS model=...`。

失败时保留 `$RUNS_ROOT/<RUN_ID>` 及完整日志，不复用旧 `RUN_ID`，不因任务失败删除现场。
