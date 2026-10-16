# 备份位置和恢复事实

## 当前发布

- 镜像：`yolo-gray1:1280-train-predict-20261016`。
- 完整 ID：`sha256:79b0086e39b83c8a3bbb76bfe2df1d8683bd443a6bef19ee7d2e69f096033635`。
- 镜像源码提交：`5b3caca445874b3567c0ad0b425c02c8d6c920e2`；后续发布提交同步说明、默认镜像与校验清单。
- GitHub 分支：`publish/h200-gray1-20261015`，未合入远端 main。
- Windows NAS：`\\10.2.26.26\902_data\0-项目\13-专项\4-代码\20261016_H200_docker_images\yolo_gray1_1280_train_predict`。
- H200 NAS：`/mnt/ywang-nas/0-项目/13-专项/4-代码/20261016_H200_docker_images/yolo_gray1_1280_train_predict`。

## 备份文件

| 文件 | 用途 |
|---|---|
| `yolo-gray1_1280-train-predict-20261016.tar` | `docker save` 导出的未压缩完整镜像，使用 `docker load` 恢复 |
| `yolo-runtime.tar.gz` | 完整构建必须的离线 Python/CUDA 用户态运行时 |
| `source.bundle`、`source.tar.gz` | 发布分支 Git 历史与发布提交的受跟踪源码快照 |
| `Dockerfile`、`BUILD_IMAGE.md`、`README.md` | 构建定义、依赖和恢复步骤 |
| `image-inspect.json`、`archive-verification.json` | 不可变镜像身份和内容校验结果 |
| `SHA256SUMS`、`VERIFIED` | 所有交付文件校验值及完成标记 |

运行时包 SHA-256：`e1e2f5eb38d40ab81c8075ef45624da04d24420e42807f78c5e9f8903e8af683`。

恢复前必须确认 `findmnt -t cifs -T "$NAS_ROOT"` 对应 `//10.2.26.26/902_data`，检查 `VERIFIED`，并在备份根目录执行 `sha256sum -c SHA256SUMS`。没有完成标记或含 `.partial` 的目录不能作为删除依据。

历史 `4-代码/训练平台/yolo-gray1-train/` 属于旧 v1 备份；当前完整构建不依赖旧 v1 镜像或三通道预训练权重。当前备份不包含数据集、正式模型权重或量化 API/Web 环境，不能代表整套平台已完成灾难恢复。

面向操作者的恢复步骤见 [环境恢复](../human/04_RESTORE_ENVIRONMENT.md)，依赖文件和打包步骤见 [完整镜像打包](../human/05_BUILD_IMAGE.md)。
