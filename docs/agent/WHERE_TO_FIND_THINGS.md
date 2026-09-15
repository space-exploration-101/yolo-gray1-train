# 文件与事实位置

收到“在哪里、怎么查、哪个生效”的问题时先用本表，不要全仓库扫描。

| 要找的内容 | 权威路径 |
|---|---|
| 人类阅读入口 | `README.md` |
| Agent 路由和约束 | `AGENT.md` |
| 标准训练烟测封装 | `run-gray1-training-smoke.sh` |
| 训练入口和参数校验 | `src/Yolo_fine/train_yolov8x_p6_pose_gray1.py` |
| Gray1 模型结构 | `src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml` |
| Ultralytics 默认参数 | `src/Yolo_fine/ultralytics/ultralytics/cfg/default.yaml` |
| Validator 单通道适配 | `src/Yolo_fine/ultralytics/ultralytics/engine/validator.py` |
| 图像转 tensor 的连续化适配 | `src/Yolo_fine/ultralytics/ultralytics/data/augment.py` |
| Docker 构建定义 | `docker/Dockerfile` |
| Python 依赖锁定 | `docker/requirements.lock.txt` |
| 运行时版本记录 | `docs/agent/runtime-packages-ultralytics-8.3.98-v1.json` |
| 发布目录校验和 | `RELEASE_SHA256SUMS` |
| 人类端到端说明 | `docs/human/01_DATA_PREPARATION.md` 至 `04_RESTORE_ENVIRONMENT.md` |
| 数据预处理实现 | `https://github.com/space-exploration-101/yolo-gray1-data-pipeline`（独立仓库） |
| 正式大文件备份 | 见 `docs/agent/BACKUP_LOCATIONS.md` |

优先执行窄范围查询：

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git status --short
rg -n '<parameter-or-symbol>' \
  run-gray1-training-smoke.sh \
  src/Yolo_fine/train_yolov8x_p6_pose_gray1.py \
  src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml \
  src/Yolo_fine/ultralytics/ultralytics/cfg/default.yaml
```

若 H200 没有 `rg`，使用同样限定路径的 `grep -nE`，不要直接递归扫描 `src/Yolo_fine/ultralytics/docs`。
