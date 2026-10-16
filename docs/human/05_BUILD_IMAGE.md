# 完整镜像打包与依赖文件

## 1. 构建入口与产物

统一使用仓库根目录下的 `docker/Dockerfile`，完整打包 1280×1280 单通道灰度图训练、验证和 PT 推理环境。

- 镜像：`yolo-gray1:1280-train-predict-20261016`。
- 已验证镜像 ID：`sha256:79b0086e39b83c8a3bbb76bfe2df1d8683bd443a6bef19ee7d2e69f096033635`。
- 构建源码：`5b3caca445874b3567c0ad0b425c02c8d6c920e2`；本次发布仅继续同步文档和默认入口。
- `Dockerfile.predictor-gray1` 是历史增量修复定义，依赖旧 v1，不作为当前构建入口。

## 2. 必备材料

| 材料 | 来源与要求 |
|---|---|
| `docker/Dockerfile`、`.dockerignore` | 当前 Git 发布分支；构建上下文必须是仓库根目录 |
| `src/Yolo_fine/` | 同一提交的完整源码，包含 Gray1 模型 YAML、训练入口和修复后的本地 Ultralytics predictor.py |
| `docker/yolo-runtime.tar.gz` | Git 忽略的大文件；从下述 NAS 根目录的 `yolo-runtime.tar.gz` 恢复 |
| `docker/yolo-runtime.tar.gz.sha256` | 运行时包预期 SHA-256；也可与下述固定值核对 |
| `ubuntu:22.04` | Dockerfile 两个阶段的基础镜像；离线构建前本机必须已有该镜像 |

NAS：`\\10.2.26.26\902_data\0-项目\13-专项\4-代码\20261016_H200_docker_images\yolo_gray1_1280_train_predict`。

运行时包 SHA-256：`e1e2f5eb38d40ab81c8075ef45624da04d24420e42807f78c5e9f8903e8af683`。

该包解压后包含 Python 3.12.12、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、OpenCV 4.11.0、NumPy 1.26.4、Albumentations 2.0.8、Pillow 12.2.0、PyYAML 6.0.3 和 tqdm 4.67.3 等运行依赖。`docker/requirements.lock.txt` 与 `docs/agent/runtime-packages-ultralytics-8.3.98-v1.json` 是依赖记录，不能替代该离线包。

2026-10-16 构建使用的基础镜像 ID 为 `sha256:714d1a2f39e9830a7c0453dc924ddf849be74811c13e4afaa21c0b9b6ad74af0`。Dockerfile 仍使用可变的 `ubuntu:22.04` 标签；再次构建不保证得到相同 image ID。离线直接恢复已保存镜像无需重新准备 Ubuntu 基础镜像。

## 3. 校验和完整构建

```bash
set -euo pipefail
NAS_ROOT='/mnt/ywang-nas/0-项目/13-专项/4-代码/20261016_H200_docker_images/yolo_gray1_1280_train_predict'
findmnt -t cifs -T "$NAS_ROOT" -o SOURCE,FSTYPE
test -r "$NAS_ROOT/VERIFIED"
(cd "$NAS_ROOT" && sha256sum -c SHA256SUMS)
test ! -e docker/yolo-runtime.tar.gz
cp "$NAS_ROOT/yolo-runtime.tar.gz" docker/yolo-runtime.tar.gz.partial
printf '%s  %s\n' \
  e1e2f5eb38d40ab81c8075ef45624da04d24420e42807f78c5e9f8903e8af683 \
  docker/yolo-runtime.tar.gz.partial | sha256sum -c -
gzip -t docker/yolo-runtime.tar.gz.partial
mv docker/yolo-runtime.tar.gz.partial docker/yolo-runtime.tar.gz
docker image inspect ubuntu:22.04 >/dev/null
docker build --network=none --pull=false -f docker/Dockerfile \
  -t yolo-gray1:1280-train-predict-candidate .
```

如果本地已存在运行时包，直接校验原包，不覆盖它。构建无需 GPU，不联网安装 Python 包；Docker 本身解析基础镜像的行为仍受本机缓存和 builder 配置影响。训练数据集、业务模型权重、运行输出不打入镜像，使用时按需挂载。该模型从单通道结构初始化训练，三通道预训练权重不是完整构建依赖。

## 4. 验证与保存

先运行训练入口 `--help` 和 CPU 推理；申请空闲 GPU 后执行单轮训练、保存 checkpoint、验证集评估和单图 GPU 推理。检查实际输入及 warmup 均为单通道，不能只看 CUDA 可用标志。2026-10-16 已通过 8 张 train/4 张 val、batch=2、1 epoch、CPU 和 GPU 推理烟测，GPU 训练峰值分配 7,577,967,104 bytes；这不是精度验收。

保存已验收的正式镜像：

```bash
docker save --output yolo-gray1_1280-train-predict-20261016.tar.partial \
  yolo-gray1:1280-train-predict-20261016
tar -tf yolo-gray1_1280-train-predict-20261016.tar.partial >/dev/null
sha256sum yolo-gray1_1280-train-predict-20261016.tar.partial
```

完成内容校验、复制后独立回读 SHA-256、同目录原子改名，再写备份清单和 `VERIFIED`。恢复时使用 `docker load --input <tar>`。本次 NAS 备份另外校验镜像配置 ID 和每个层的内容摘要，并从 NAS 执行 docker load；这不等于新主机 clean-room 恢复。

返回：[项目首页](../../README.md)
