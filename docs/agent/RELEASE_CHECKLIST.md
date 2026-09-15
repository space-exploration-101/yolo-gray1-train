# 提交和推送前必须同步相关文档

用户要求提交或更新远程仓库时，Agent 必须主动执行本清单，不等待额外提醒。

## 先按改动定位文档

```bash
REPO_ROOT=$(git rev-parse --show-toplevel)
cd "$REPO_ROOT"
git status --short
git diff --name-only
git diff --cached --name-only
```

| 改动范围 | 同步检查 |
|---|---|
| 数据契约或预处理接口 | `docs/human/01_DATA_PREPARATION.md`、`docs/agent/DATASET_CONTRACT.md` |
| 训练脚本、GPU 门禁、输出 | `docs/human/02_RUN_TRAINING.md`、`docs/agent/TRAINING_INTERFACE.md` |
| 参数、模型 YAML、依赖 | `docs/human/03_TRAINING_CONFIGURATION.md`、`docs/agent/TRAINING_INTERFACE.md`、运行时 JSON |
| Docker、镜像、NAS 或恢复 | `docs/human/04_RESTORE_ENVIRONMENT.md`、`docs/agent/BACKUP_LOCATIONS.md` |
| 新增、移动或删除主要入口 | `README.md`、`AGENT.md`、`docs/agent/WHERE_TO_FIND_THINGS.md` |
| 任意发布文件变化 | `RELEASE_SHA256SUMS` |

## 推送门禁

1. 更新上述相关文档；检查 human 文档仍按数字顺序可执行，agent 文档仍能直接按主题定位。
2. 更新 `RELEASE_SHA256SUMS` 中新增、修改、移动或删除文件的条目。
3. 校验 Markdown 相对链接，确认没有指向旧路径。
4. 运行 `git diff --cached --check`；上游导入文件的既有 CRLF/行尾问题可记录，但不得借文档提交批量改写已验收源码。
5. 确认没有跟踪超过 100 MB 的文件，没有 `.partial`、私钥、token、数据集、训练输出或 Docker 归档。
6. 查看 `git diff --cached --stat` 和关键 diff，使用仓库本地作者信息提交。
7. 推送前先 `git fetch`，确认远端没有未处理的新提交；禁止 force push。
8. 推送后比较本地 `HEAD` 与远端 `main`，并确认工作树干净。

只更新文档或 Git 不需要 GPU、训练或服务重启。若代码改动影响运行行为，应按风险执行对应静态检查或受控烟测，并在结果中明确说明是否实际使用 GPU。
