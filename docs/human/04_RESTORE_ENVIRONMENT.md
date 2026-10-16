# 环境恢复

GitHub 保存代码和小型配置；NAS 保存完整镜像与离线运行时。当前训练和推理共用 `yolo-gray1:1280-train-predict-20261016`，完整 ID 为 `sha256:79b0086e39b83c8a3bbb76bfe2df1d8683bd443a6bef19ee7d2e69f096033635`。

## 1. 检查备份

Windows NAS：

```text
\\10.2.26.26\902_data\0-项目\13-专项\4-代码\20261016_H200_docker_images\yolo_gray1_1280_train_predict
```

在 Linux 上使用已授权挂载，先检查真实 CIFS 来源：

```bash
set -euo pipefail
NAS_ROOT='/mnt/ywang-nas/0-项目/13-专项/4-代码/20261016_H200_docker_images/yolo_gray1_1280_train_predict'
findmnt -t cifs -T "$NAS_ROOT" -o TARGET,SOURCE,FSTYPE
test -r "$NAS_ROOT/VERIFIED"
(cd "$NAS_ROOT" && sha256sum -c SHA256SUMS)
test -z "$(find "$NAS_ROOT" -name '*.partial' -print)"
```

来源必须是 `//10.2.26.26/902_data`，校验必须全部通过。

## 2. 导入完整镜像

先核对同名标签，只有标签不存在或指向以下已知 ID 时才继续：

```bash
IMAGE='yolo-gray1:1280-train-predict-20261016'
EXPECTED='sha256:79b0086e39b83c8a3bbb76bfe2df1d8683bd443a6bef19ee7d2e69f096033635'
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$EXPECTED"
fi
docker load --input "$NAS_ROOT/yolo-gray1_1280-train-predict-20261016.tar"
test "$(docker image inspect "$IMAGE" --format '{{.Id}}')" = "$EXPECTED"
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=1g \
  -e HOME=/tmp -e NO_ALBUMENTATIONS_UPDATE=1 "$IMAGE" \
  python /workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py --help
```

这是未压缩 Docker TAR，使用 `docker load`，不能使用 `docker import`。镜像内已包含运行时和训练、推理源码；运行时不需要再挂载离线包或宿主源码。数据、模型权重及输出目录需单独提供；GPU 使用仍需宿主驱动、NVIDIA Container Toolkit 和资源门禁。

## 3. 恢复源码与重建

在新的空目录 clone 发布分支：

```bash
git clone --branch publish/h200-gray1-20261015 \
  https://github.com/space-exploration-101/yolo-gray1-train.git yolo-gray1-train-restored
cd yolo-gray1-train-restored
```

以 NAS `source-state.json` 中的发布提交为准固定 checkout；离线时可从 `source.bundle` clone 后 checkout 同一提交。恢复运行时包到 `docker/yolo-runtime.tar.gz`，再按照 [完整镜像打包与依赖](05_BUILD_IMAGE.md) 执行。完整构建不需要旧 v1 镜像，也不要求三通道预训练 `.pt`。

## 4. 恢复后验收

数据集和正式权重另行恢复并核对各自 SHA-256，不能用备份中的烟测权重代替正式模型。GPU 使用前持续采样显存、利用率和进程所有者，再按 [训练烟测](02_RUN_TRAINING.md) 执行。2026-10-16 的 8 张 train/4 张 val、1 epoch 烟测及 CPU/GPU 推理已通过，仅验证功能，不代表完整模型精度验收。

返回：[项目首页](../../README.md)
