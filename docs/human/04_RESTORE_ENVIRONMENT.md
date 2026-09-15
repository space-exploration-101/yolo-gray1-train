# 环境恢复

本文是端到端复现的第 4 步，也是新主机或环境丢失时的独立恢复手册。GitHub 保存源码和小文件；NAS 保存 Docker 镜像归档、离线运行时及未进入 Git 的大权重。恢复过程不需要 GPU，但会写入本地目录并可能导入 Docker 镜像。

## 1. 备份位置

GitHub：

```text
https://github.com/space-exploration-101/yolo-gray1-train
```

NAS：

```text
Windows:
\\10.2.26.26\902_data\0-项目\13-专项\4-代码\训练平台\yolo-gray1-train

H200：NAS 的实际挂载点由管理员或当前用户决定，挂载后将其赋给 `NAS_MOUNT`。
```

NAS 备份包含：

- `docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst`：正式训练镜像。
- `docker/yolo-runtime.tar.gz`：Dockerfile 重建所需的离线运行时。
- `src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt`：未进入 Git 的大权重。
- `SHA256SUMS`、`VERIFIED`：备份校验和与完成标记。

## 2. 恢复前门禁

确认 NAS 确实挂载、可读且目标备份完成：

```bash
ssh '<h200-host>'

WORK_ROOT='<absolute-writable-workspace>'
NAS_MOUNT='<absolute-mounted-902_data-path>'
NAS_ROOT="$NAS_MOUNT/0-项目/13-专项/4-代码/训练平台/yolo-gray1-train"
findmnt -T "$NAS_MOUNT"
test -r "$NAS_ROOT/VERIFIED"
test -r "$NAS_ROOT/SHA256SUMS"
df -h "$WORK_ROOT" "$NAS_MOUNT"
```

如果 `findmnt` 没有显示预期 SMB 文件系统，应先按主机运维流程挂载 NAS；不要把空的本地挂载点误当成 NAS。恢复到新目录，目标已存在时停止，不覆盖旧内容。

## 3. 校验 NAS 备份

```bash
cd "$NAS_ROOT"
sha256sum -c SHA256SUMS
zstd -q -t docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst
gzip -t docker/yolo-runtime.tar.gz
find . -name '*.partial' -print
```

所有校验必须通过，最后一条命令必须没有输出。

## 4. 恢复源码

选择一个不存在的新目录：

```bash
RESTORE_ROOT="$WORK_ROOT/yolo-gray1-train-restored"
test ! -e "$RESTORE_ROOT"
git clone https://github.com/space-exploration-101/yolo-gray1-train.git "$RESTORE_ROOT"
cd "$RESTORE_ROOT"
```

若使用 deploy key，可将仓库 URL 换成组织批准的 SSH 地址。不要把私钥路径或内容写入仓库。

## 5. 优先恢复正式镜像

先检查同名镜像是否已经存在：

```bash
TRAIN_IMAGE='ywang/yolo-gray1-train:ultralytics-8.3.98-v1'
docker image inspect "$TRAIN_IMAGE" >/dev/null 2>&1 \
  && echo 'image already exists; compare identity before loading' \
  || echo 'image tag is free'
```

确认不会覆盖需要保留的同名镜像后再导入：

```bash
zstd -dc "$NAS_ROOT/docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst" \
  | docker load
docker image inspect "$TRAIN_IMAGE"
```

## 6. 需要重建镜像时

先将 Git 忽略的大文件恢复到源码树；使用临时文件、校验后再原子改名：

```bash
cp "$NAS_ROOT/docker/yolo-runtime.tar.gz" docker/yolo-runtime.tar.gz.partial
sha256sum docker/yolo-runtime.tar.gz.partial
test "$(sha256sum docker/yolo-runtime.tar.gz.partial | awk '{print $1}')" = \
  "$(cut -d' ' -f1 docker/yolo-runtime.tar.gz.sha256)"
mv docker/yolo-runtime.tar.gz.partial docker/yolo-runtime.tar.gz

cp "$NAS_ROOT/src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt" \
  src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt.partial
test "$(sha256sum src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt.partial | awk '{print $1}')" = \
  "$(grep 'yolov8x-pose-p6.pt$' "$NAS_ROOT/SHA256SUMS" | awk '{print $1}')"
mv src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt.partial \
  src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt
```

构建时建议先用临时标签，验证后再决定是否赋予正式标签：

```bash
docker build \
  -f docker/Dockerfile \
  -t ywang/yolo-gray1-train:recovery-candidate .

docker run --rm \
  --network none \
  --read-only \
  ywang/yolo-gray1-train:recovery-candidate \
  python /workspace/src/Yolo_fine/train_yolov8x_p6_pose_gray1.py --help
```

## 7. 发布目录校验与验收

大文件均已恢复到原相对路径后：

```bash
cd "$RESTORE_ROOT"
sha256sum -c RELEASE_SHA256SUMS
```

静态校验和镜像检查通过后，再按照[启动并验证训练](02_RUN_TRAINING.md)执行一次单卡 1 epoch 烟测。只有出现 `TRAINING_SMOKE_PASS`，才说明数据、GPU、训练、checkpoint 和 final evaluation 链路均已贯通。

返回：[项目首页](../../README.md)
