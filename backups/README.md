# 集成快照目录

每次功能集成完成后，把当时的 **`app.py` 主入口** 备份到这里。项目根目录始终保留最新可运行版本，历史里程碑存快照。

> 快照索引见 [`manifest.json`](manifest.json)。

---

## 快速上手

```bash
cd personal_assistant

# 1. 启动服务（始终用最新版 app.py）
uv run app.py

# 2. 功能集成完成后，创建快照
uv run python scripts/snapshot.py <阶段标签> "<本次集成说明>"

# 3. 查看已有快照
cat backups/manifest.json
ls backups/snapshots/
```

**示例**：

```bash
uv run python scripts/snapshot.py stage62_agent_graph "P1 /chat/graph 接入 + app.py 重命名"
# → backups/snapshots/20260621_stage62_agent_graph/app.py
```

---

## 命名规范

```
backups/snapshots/YYYYMMDD_<阶段标签>/
```

| 部分 | 说明 | 示例 |
|------|------|------|
| `YYYYMMDD` | 快照日期（脚本自动加） | `20260621` |
| `<阶段标签>` | 你起的短名，英文+下划线 | `stage62_agent_graph` |

**标签建议**：`stage<N>_<功能>`，如 `stage62_web_search`、`stage63_mcp_tools`。

---

## 完整工作流

```
改 app.py
    ↓
本地验证（uv run app.py + 浏览器/curl）
    ↓
功能集成完毕 → 跑 snapshot.py 备份 app.py
    ↓
继续下一功能（根目录始终是最新版）
```

### 创建快照

```bash
uv run python scripts/snapshot.py <阶段标签> "<说明>"
```

| 参数 | 必填 | 说明 |
|------|------|------|
| `阶段标签` | ✅ | 如 `stage62_web_search` |
| `"说明"` | 可选 | 一句话描述本次集成内容 |

### 查看快照内容

每个快照目录只含两个文件：

```
20260621_stage62_agent_graph/
├── MANIFEST.md          # 本次快照说明
└── app.py               # 当时的入口
```

### 恢复某个版本

```bash
cp backups/snapshots/20260621_stage62_agent_graph/app.py app.py
```

恢复后重启服务：`uv run app.py`

### 查看全部快照索引

```bash
cat backups/manifest.json
```

---

## 与 Git 的关系

| | Git | 快照 |
|---|-----|------|
| 粒度 | commit（任意 diff） | 集成里程碑（app.py 副本） |
| 恢复 | `git checkout` / `git restore` | `cp` 从 snapshots 目录 |
| 适合 | 日常开发、协作、历史追溯 | 「这个功能完成时 app.py 长什么样」 |

**建议习惯**：重要集成 = **先 snapshot，再 git commit**。

---

## 主入口命名说明

| 文件 | 状态 | 说明 |
|------|------|------|
| `app.py` | ✅ 正式入口 | 启动：`uv run app.py` |
| `step23_rag.py` | 兼容壳 | 旧命令仍可用，会提示并转发到 `app.py` |

历史快照里可能仍叫 `step23_rag.py`（P1 之前），以快照目录内文件名为准。

---

## 已有快照

| 名称 | 说明 |
|------|------|
| `20260621_stage62_agent_graph` | P1 `/chat/graph` 接入 + `app.py` 重命名 |

完整列表见 [`manifest.json`](manifest.json)。

---

## 常见问题

**Q：同一天集成两次，label 重复怎么办？**  
脚本会报错「快照已存在」。换 label，如 `stage62_agent_graph_v2`。

**Q：快照占空间吗？**  
每次只存一个 `app.py`（约几百 KB），体积很小。旧快照可手动删目录 + 编辑 `manifest.json`。

**Q：为什么不备份 agent/、前端等？**  
那些文件用 Git 管理；快照只留主入口里程碑，避免占空间。

**Q：恢复后服务起不来？**  
检查 `.env`、Docker 4 容器是否在跑，命令用 `uv run app.py`。

---

## 相关文件

| 路径 | 作用 |
|------|------|
| `scripts/snapshot.py` | 快照创建脚本 |
| `backups/manifest.json` | 全部快照索引 |
| `backups/snapshots/` | 快照存放目录 |
