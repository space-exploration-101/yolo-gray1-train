# 数据预处理

本文说明当前已经在 H200 上验证的有界烟测预处理流程。命令均在 H200 宿主机执行，不需要 GPU；仓库、原始数据和输出目录由操作者自行选择，不依赖特定用户名。

## 1. 输入与输出契约

数据预处理实现位于独立仓库：[space-exploration-101/yolo-gray1-data-pipeline](https://github.com/space-exploration-101/yolo-gray1-data-pipeline)。该仓库不包含业务数据或完整数据集；操作者必须提供自己的原始数据目录，并根据仓库说明明确源图语义（例如 `native_gray`、`r_only` 或 `color_to_gray`）。预处理会输出真正的单通道 `gray1` 图像。

此前端到端验证使用的是 21 类合成烟测数据，不是业务生产数据。它只证明 H200 上的软件链路能够贯通，不应作为新用户必须拥有的固定路径或数据来源。

训练输入必须是已经严格校验的 prepared dataset：

```text
<prepared-dataset>/
├── VERIFIED
├── dataset.yaml
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

量化打包还会使用 `images/calibration/`、`images/test/` 和 `labels/test/`。`dataset.yaml` 必须声明：

```yaml
channels: 1
input_semantics: gray1
kpt_shape: [2, 3]
```

类别名称必须恰好有 21 个，不得启用旧的 `r_channel_only` 模式。`VERIFIED` 必须由校验程序生成，不能手工创建空文件代替。

## 2. 创建独立运行目录

每次运行使用新目录，禁止覆盖已有结果：

```bash
ssh '<h200-host>'

WORK_ROOT='<absolute-writable-workspace>'
RUNS_ROOT='<absolute-writable-runs-directory>'
RAW='<absolute-raw-dataset-directory>'
RUN_ID="prep_$(date +%Y%m%d_%H%M%S)"
GRAYPREP_REPO="$WORK_ROOT/yolo-gray1-data-pipeline"
PREP_RUN="$RUNS_ROOT/$RUN_ID"

test -d "$WORK_ROOT"
test -d "$RUNS_ROOT"
test -d "$RAW"
test ! -e "$GRAYPREP_REPO"
git clone https://github.com/space-exploration-101/yolo-gray1-data-pipeline.git \
  "$GRAYPREP_REPO"
test -d "$GRAYPREP_REPO"
test ! -e "$PREP_RUN"
mkdir -p "$PREP_RUN"
```

如果仓库已经克隆，不要再次执行 `git clone`，直接把 `GRAYPREP_REPO` 指向现有目录。以上变量只在当前 shell 中有效；后续命令必须在同一个终端执行，重新登录后需要重新设置。

## 3. 确定性选择烟测样本

下面命令按固定种子从每个类别选择 1 个样本，生成 `selection.json`：

```bash
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  -e PYTHONPATH=/app/src \
  -v "$GRAYPREP_REPO/src:/app/src:ro" \
  -v "$GRAYPREP_REPO/configs:/app/configs:ro" \
  -v "$GRAYPREP_REPO/schemas:/app/schemas:ro" \
  -v "$RAW:/source:ro" \
  -v "$PREP_RUN:/output:rw" \
  ywang/yolo-gray1-data-pipeline:0.1.1 \
  dataset select-smoke \
  --source /source \
  --output /output/selection.json \
  --seed 20260912 \
  --per-class 1

test -s "$PREP_RUN/selection.json"
```

`--seed` 保证选择可复现；`--per-class` 控制每类样本数量。当前命令用于烟测，不代表全量数据选择策略。

## 4. 构建 prepared dataset

`net1280` 将图像缩放到 `1000×1000`，四周补 140 像素后形成 `1280×1280 gray1`，并同步变换框和关键点。`cam2000` 产物用于软件侧相机 BIN 测试。

```bash
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  -e PYTHONPATH=/app/src \
  -v "$GRAYPREP_REPO/src:/app/src:ro" \
  -v "$GRAYPREP_REPO/configs:/app/configs:ro" \
  -v "$GRAYPREP_REPO/schemas:/app/schemas:ro" \
  -v "$RAW:/source:ro" \
  -v "$PREP_RUN:/output:rw" \
  ywang/yolo-gray1-data-pipeline:0.1.1 \
  dataset build-smoke \
  --source /source \
  --manifest /output/selection.json \
  --output /output/prepared \
  --net-config /app/configs/net1280.yaml \
  --cam-config /app/configs/cam2000.yaml \
  --schema /app/schemas/preprocess-profile.schema.json \
  --workers 32 \
  --opencv-threads 1
```

已验证的并行设置是 32 个 worker、每进程 1 个 OpenCV 线程。共享服务器繁忙时应降低 `--workers`。大量数据的总耗时主要取决于图像数量和存储吞吐，正式全量处理前应先用代表性子集测量速度和预计容量。

## 5. 严格校验

校验会重新读取产物，检查清单、文件配对、图像/标签约束、变换一致性和内容哈希；因此会产生与数据量近似线性增长的 I/O 和 CPU 开销。

```bash
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  -e PYTHONPATH=/app/src \
  -v "$GRAYPREP_REPO/src:/app/src:ro" \
  -v "$PREP_RUN:/output:rw" \
  -v "$RAW:/source:ro" \
  ywang/yolo-gray1-data-pipeline:0.1.1 \
  dataset verify-smoke \
  --output /output/prepared \
  --source /source

test -f "$PREP_RUN/prepared/VERIFIED"
DATASET="$PREP_RUN/prepared"
```

若提示 `/output/selection.json` 不存在，说明第 3 步未在当前 `PREP_RUN` 中成功完成，或 shell 变量已改变。先用 `printf` 和 `ls` 检查变量及文件，不要在其他目录盲目重跑：

```bash
printf 'GRAYPREP_REPO=%s\nRAW=%s\nPREP_RUN=%s\n' \
  "$GRAYPREP_REPO" "$RAW" "$PREP_RUN"
ls -lah "$PREP_RUN"
```

返回：[项目首页](../../README.md) · 下一步：[启动并验证训练](02_RUN_TRAINING.md)
