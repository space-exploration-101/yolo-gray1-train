# Gray1 数据集必须满足的契约

训练入口会读取 `dataset.yaml` 并强制检查以下字段：

```yaml
channels: 1
input_semantics: gray1
kpt_shape: [2, 3]
names: [...]            # 恰好 21 项
```

同时要求：

- `r_channel_only` 不存在或为 `false`。
- 数据根目录有预处理验证程序生成的 `VERIFIED`。
- `images/train`、`images/val`、`labels/train`、`labels/val` 存在且图像/标签同名配对。
- 图像能按真正的单通道灰度图读取，不是 `R=灰度,G=B=0` 的三通道表示。
- 量化输入打包还依赖 `images/calibration`、`images/test` 和 `labels/test`。

对应代码与配置：

| 事实 | 位置 |
|---|---|
| 数据 YAML 校验 | `src/Yolo_fine/train_yolov8x_p6_pose_gray1.py::validate_args` |
| 模型通道、类别和关键点 | `src/Yolo_fine/configs/yolov8x-pose-p6-gray1.yaml` |
| 预处理配置和实现 | `https://github.com/space-exploration-101/yolo-gray1-data-pipeline` |
| 人类操作步骤 | `docs/human/01_DATA_PREPARATION.md` |

本仓库不附带默认数据集。已验证环境曾使用 21 类合成测试数据贯通 H200 软件链路，但该数据不是业务生产数据，也不是可移植接口的一部分。回答数据来源问题时应要求用户提供或确认当前 `DATASET`，不要从历史路径推断。

最小只读检查：

```bash
DATASET='<prepared-dataset>'
test -f "$DATASET/VERIFIED"
test -f "$DATASET/dataset.yaml"
sed -n '1,120p' "$DATASET/dataset.yaml"
find "$DATASET" -maxdepth 2 -type d -print
```
