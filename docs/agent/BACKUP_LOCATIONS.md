# 备份位置和恢复事实

## Git 与 NAS 分工

| 内容 | 位置 |
|---|---|
| 源码、小配置和说明 | `https://github.com/space-exploration-101/yolo-gray1-train` |
| H200 仓库 | 任意用户可写位置；用 `git rev-parse --show-toplevel` 获取 |
| H200 NAS 备份 | `$NAS_MOUNT/0-项目/13-专项/4-代码/训练平台/yolo-gray1-train` |
| Windows NAS 路径 | `\\10.2.26.26\902_data\0-项目\13-专项\4-代码\训练平台\yolo-gray1-train` |

NAS 中的正式大文件：

| 文件 | SHA-256 |
|---|---|
| `docker/yolo-runtime.tar.gz` | `e1e2f5eb38d40ab81c8075ef45624da04d24420e42807f78c5e9f8903e8af683` |
| `docker/yolo-gray1-train-ultralytics-8.3.98-v1.tar.zst` | `6a35e9cf05fa90547ab176125c363257e8dc5a69d2012a49cd149e8134b0ce3b` |
| `src/Yolo_fine/ultralytics/yolov8x-pose-p6.pt` | `319004a4a2fe2da6734f881fe628f5825e5864f59a990e0faa90f7cbe30f4842` |

恢复前只读门禁：

```bash
NAS_MOUNT='<absolute-mounted-902_data-path>'
NAS_ROOT="$NAS_MOUNT/0-项目/13-专项/4-代码/训练平台/yolo-gray1-train"
findmnt -T "$NAS_MOUNT"
test -r "$NAS_ROOT/VERIFIED"
cd "$NAS_ROOT"
sha256sum -c SHA256SUMS
find . -name '*.partial' -print
```

不要把未挂载的空目录当成 NAS。不要覆盖已有恢复目录或同名镜像；先检查目标状态。面向操作者的完整恢复步骤位于 `docs/human/04_RESTORE_ENVIRONMENT.md`。
