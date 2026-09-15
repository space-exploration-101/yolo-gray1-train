# Agent 入口

本文件是自动化 Agent 在本仓库的第一读取入口。目标是用最少读取快速定位事实；不要先扫描整个仓库，也不要把历史提交或人类教程当作当前操作约束。

## 最小读取规则

1. 先读本文件。
2. 根据用户问题只读下表对应的主题文档和目标源码。
3. 只有主题文档明确要求时才扩大搜索范围。
4. 修改前检查 `git status --short`，保留并避开未知改动。
5. GPU、训练、Docker 导入、NAS 写入和删除操作必须分别确认授权；H200 是共享服务器。

## 按问题定位

| 用户问题或任务 | 首先读取 |
|---|---|
| “文件在哪里”“实现在哪一层” | [`docs/agent/WHERE_TO_FIND_THINGS.md`](docs/agent/WHERE_TO_FIND_THINGS.md) |
| 数据格式、Gray1、类别或关键点契约 | [`docs/agent/DATASET_CONTRACT.md`](docs/agent/DATASET_CONTRACT.md) |
| 如何启动、GPU 门禁、输出和成功条件 | [`docs/agent/TRAINING_INTERFACE.md`](docs/agent/TRAINING_INTERFACE.md) |
| 参数来源、默认值、固定值或实际生效值 | [`docs/agent/TRAINING_INTERFACE.md`](docs/agent/TRAINING_INTERFACE.md) |
| 镜像、NAS、备份位置或恢复 | [`docs/agent/BACKUP_LOCATIONS.md`](docs/agent/BACKUP_LOCATIONS.md) |
| 修改、提交或同步 GitHub | [`docs/agent/RELEASE_CHECKLIST.md`](docs/agent/RELEASE_CHECKLIST.md) |
| 面向操作者的端到端步骤 | [`README.md`](README.md)，再按数字阅读 `docs/human/` |

## 当前硬约束

- 已验证运行环境：NVIDIA H200；仓库位置使用 `REPO_ROOT=$(git rev-parse --show-toplevel)` 获取
- 正式镜像：`ywang/yolo-gray1-train:ultralytics-8.3.98-v1`
- 模型输入：`gray1 [B,1,1280,1280]`
- 模型契约：`ch: 1`、`nc: 21`、`kpt_shape: [2, 3]`
- 标准烟测：单张空闲 GPU、1 epoch、55 分钟硬超时
- 大文件不进入 Git；从 NAS 恢复
- 不推送密钥、数据集、运行产物、`.pt`、镜像归档或 `.partial`

## 推送前必须自动同步文档

当用户要求把变更提交或推送到远程仓库时，Agent 必须在提交前执行 [`docs/agent/RELEASE_CHECKLIST.md`](docs/agent/RELEASE_CHECKLIST.md)。根据本次改动同步相关 human/agent 文档、`README.md`、`AGENT.md` 和 `RELEASE_SHA256SUMS`；不得等待用户额外提醒。
