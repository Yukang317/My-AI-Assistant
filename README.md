# 个人 AI 助理（Personal Assistant）

> 把我的知识、记忆、习惯数字化，让 AI 理解我，帮我做事。

一个运行在阿里云 ECS 上的个人 AI 助理系统，具备 RAG 知识库检索、Agent 智能编排、流式聊天对话等能力。技术栈覆盖大模型应用开发全链路，既是生产力工具，也是教学练手项目。

**核心能力路径**：Web 聊天 → 持久记忆 → RAG 知识库 → Agent 编排 → 工具生态 → 多平台接入 → 长期记忆

---

## 功能概览

| 能力 | 状态 | 说明 |
|------|------|------|
| 💬 流式聊天 | ✅ | SSE 流式对话，多会话管理，历史持久化 |
| 📚 RAG 知识库 | ✅ | 文档上传→解析→分块→向量化→混合检索→AI 增强回答 |
| 🤖 Agent 编排 | ✅ 6.1 | LangGraph 多步编排，意图路由 + 工具调用 |
| 📄 文档管理 | ✅ | 上传/列表/删除，支持 PDF/Word/Markdown/TXT |
| 🔍 混合检索 | ✅ | BM25 关键词 + 向量语义 → RRF 融合 → 可选重排 |
| 🧪 检索评估 | ✅ | RAGAS 框架，忠实度+相关度+准确度+精密度 |
| 🌐 网页搜索 | ✅ | Exa/Tavily 双搜索 API |
| 🧠 文件记忆 | 🔄 6.2 | Markdown 文件记忆系统（规划中） |

---

## 架构一览

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI (:8000)                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ 普通聊天   │  │ 文档管理 API │  │ RAG 增强聊天    │  │
│  │ /chat      │  │ /api/docs/*  │  │ /chat?use_rag   │  │
│  └─────┬──────┘  └──────┬───────┘  └────────┬────────┘  │
│        │                │                    │           │
│  ┌─────▼────────────────▼────────────────────▼────────┐  │
│  │                  Agent 编排层                        │  │
│  │  load_context → intent_route → tool_execute         │  │
│  │                        ↓                            │  │
│  │                 result_synthesis                    │  │
│  └────────────────────────┬───────────────────────────┘  │
│                           │                              │
│  ┌────────────────────────▼───────────────────────────┐  │
│  │                   RAG 检索层                         │  │
│  │  BM25 关键词 ─┐                                    │  │
│  │               ├─ RRF 融合 → 重排序 → LLM 生成      │  │
│  │  向量语义   ──┘                                    │  │
│  └──────┬─────────────────────────┬───────────────────┘  │
└─────────┼─────────────────────────┼──────────────────────┘
          │                         │
┌─────────▼─────────┐  ┌────────────▼──────────────┐
│ PostgreSQL(:15433)│  │   Milvus(:19530)           │
│  元数据 + 聊天记录 │  │   向量索引（父子双集合）   │
└───────────────────┘  └────────────┬───────────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  MinIO(:9000)        │
                         │  原始文件存储        │
                         └─────────────────────┘
```

### Docker 基础设施（4 容器协同）

| 容器 | 端口 | 用途 | 启动依赖 |
|------|------|------|----------|
| **etcd** | 2379 | 分布式协调，Milvus 元数据存储 | — |
| **MinIO** | 9000 (API) / 9001 (控制台) | S3 兼容对象存储，存原始文件 | etcd |
| **Milvus** | 19530 (gRPC) / 9091 (健康检查) | 向量数据库，父子块索引 | etcd + MinIO |
| **PostgreSQL** | 15433 | 结构化存储，元数据+聊天记录 | — |

---

## 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI + Uvicorn | SSE 流式，14 个 API 端点 |
| Agent 编排 | LangGraph | 4 节点主图 + while-true 循环 + MemorySaver |
| LLM | DeepSeek API | OpenAI 兼容，性价比高 |
| 向量数据库 | Milvus Standalone 2.4 | IVF_FLAT + IP 度量，父子双集合 |
| 对象存储 | MinIO (S3 兼容) | 存原始文档，支持分片上传 |
| 结构化存储 | PostgreSQL 17 | 聊天记录、文档元数据、会话管理 |
| Embedding | BGE-large-zh-v1.5 (1024 维) | 支持本地/API 双模式，默认 SiliconFlow API |
| 文档解析 | PyMuPDF / python-docx / markitdown | 策略模式，可扩展解析器 |
| 文档分块 | 语义分块 + 父子块 | 父块 2048/128，子块 512/64 |
| 关键词检索 | BM25 (rank-bm25) | 内存索引，从 Milvus 反查构建 |
| 重排序 | bge-reranker-large | 可选，默认关闭 |
| 检索评估 | RAGAS | 忠实度 + 相关度 + 准确度 + 精密度 |
| 前端 | 单文件 HTML/CSS/JS | ~1600 行，零框架依赖 |
| 包管理 | uv + Python 3.12 | Aliyun pip 镜像 |
| 容器化 | Docker Compose | 4 容器，healthcheck 启动编排 |

**刻意不用的**：Redis（阶段 6 后期引入）、GraphRAG（个人文档量小）。

---

## 项目结构

```
personal_assistant/
├── app.py                      # 主服务入口（FastAPI，15 API 端点）
├── backups/                    # app.py 集成快照（见下方「集成快照」）
│   ├── snapshots/              #   各里程碑的 app.py 副本
│   └── manifest.json           #   快照索引
├── scripts/snapshot.py         # 快照创建脚本（只备份 app.py）
├── db.py                       # PostgreSQL 数据库抽象层（12 函数）
├── migrate_sqlite_to_pg.py     # SQLite → PG 一次性迁移脚本
├── docker-compose.yml          # Docker 基础设施（4 容器）
├── pyproject.toml              # 项目依赖 + uv 配置
├── .env                        # 环境变量（gitignored）
│
├── rag/                        # RAG 知识库模块（~2000 行）
│   ├── config.py               #   全局配置（环境变量 + 默认值，9 类配置）
│   ├── embedding.py            #   Embedding 服务（本地 BGE / 远程 API 双模式）
│   ├── indexer.py              #   文档摄入编排（6 函数：上传→解析→分块→向量化→存储）
│   ├── rag_service.py          #   检索编排层（8 函数，检索管线闭环）
│   ├── evaluate.py             #   RAGAS 评估模块（9 函数）
│   ├── parser/                 #   文档解析器（策略模式）
│   │   ├── base.py             #     解析器基类 + 工厂注册
│   │   ├── coordinator.py      #     协调器（按类型分发→聚合结果）
│   │   ├── pdf_parser.py       #     PDF（PyMuPDF）
│   │   ├── docx_parser.py      #     Word（python-docx）
│   │   ├── markdown_parser.py  #     Markdown（markitdown）
│   │   └── text_parser.py      #     纯文本
│   ├── chunker/                #   文档分块器
│   │   ├── semantic_splitter.py #    语义分块（基于嵌入相似度判断断点）
│   │   └── parent_child_builder.py # 父子块构建器
│   ├── storage/                #   存储适配器
│   │   ├── minio_client.py     #     MinIO 对象存储客户端
│   │   └── vector_store.py     #     Milvus 向量存储（14 方法，父子双集合）
│   └── retrieval/              #   检索器
│       ├── bm25.py             #     BM25 关键词检索（内存索引）
│       ├── vector_retriever.py #     向量语义检索
│       └── rrf_fusion.py       #     RRF 融合排序
│
├── agent/                      # Agent 编排模块（~1200 行，阶段 6.1 ✅）
│   ├── state.py                #   主状态定义（TypedDict + 自定义 reducer）
│   ├── main_graph.py           #   主图构建（4 节点 + 条件边 + MemorySaver）
│   ├── router.py               #   LLM 意图路由（general_chat / use_tool）
│   ├── bound.py                #   BOUND 约束层（安全边界）
│   ├── nodes/                  #   图节点实现
│   │   ├── load_context.py     #     上下文加载（PG 历史 + 文件记忆）
│   │   └── result_synthesis.py #     结果合成（模板注入 + 防 prompt injection）
│   └── tools/                  #   工具系统（可插拔注册表）
│       ├── base.py             #     ToolContext + ToolResult 数据类
│       ├── registry.py         #     工具注册表（register / get / list）
│       ├── rag_search.py       #     RAG 搜索工具
│       └── web_search.py       #     网页搜索工具（Exa + Tavily）
│
├── static/
│   └── index.html              # 聊天前端（~1600 行，单文件）
│
├── learn_basis/                # 学习示例
│   └── step01_single_tool_agent.py  # 单工具 Agent 教学示例
│
├── docs/                       # 项目文档
│   ├── 上次会话成果.md           # ★ 最新进度与代码状态（每次会话后更新）
│   ├── 阶段5技术方案.md          # RAG 完整架构方案（按需）
│   └── 阶段6技术方案.md          # Agent 编排架构方案（按需）
│
├── dev_log/                    # 开发日志（按日期归档）
└── 备份/                       # 旧版本代码备份
```

---

## 三存储系统协作

项目使用三个存储系统各司其职，通过 `doc_id` 关联：

| 存储 | 类比 | 存什么 | 特点 |
|------|------|--------|------|
| **MinIO** | 文件柜 | 原始文档（PDF/Word/MD/TXT） | 不可变，一次上传永久保存 |
| **Milvus** | 搜索引擎 | 向量嵌入 + chunk 文本 | 语义相似度搜索，父子双集合 |
| **PostgreSQL** | 档案卡片柜 | 元数据 + 聊天记录 + 父子映射 | 结构化查询，ACID 事务 |

删除时按顺序清理：BM25 内存先删 → MinIO+Milvus 持久层 → PG 元数据最后。

---

## 快速开始

### 运行环境

项目部署在阿里云 ECS（Alibaba Cloud Linux 3，3.5GB RAM，80GB 磁盘），Python 3.12.12（uv 管理），pip 使用 Aliyun 镜像。

### 1. 准备 Python 环境

```bash
# 确保 uv 已安装
pip install uv

# 安装项目依赖
cd personal_assistant
uv sync
```

### 2. 启动基础设施（Docker）

```bash
# 启动 4 个容器（etcd → MinIO → Milvus + PostgreSQL）
docker compose up -d

# 确认所有容器 healthy
docker compose ps
```

### 3. 配置环境变量

参考 `rag/config.py` 中的默认值，创建 `.env` 文件覆盖必要变量：

```bash
# 最少需要配置
DEEPSEEK_API_KEY=your_deepseek_api_key
EMBEDDING_API_KEY=your_siliconflow_api_key    # SiliconFlow 提供免费额度
```

### 4. 启动服务

```bash
uv run app.py
```

服务启动后：
- API 地址：`http://0.0.0.0:8000`
- 聊天前端：`http://localhost:8000/static/index.html`
- MinIO 控制台：`http://localhost:9001`（minioadmin / minioadmin）

---

## API 端点一览

### 会话管理
| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/sessions` | 获取所有会话列表 |
| `POST` | `/sessions` | 创建新会话 |
| `PUT` | `/sessions/{id}/rename` | 重命名会话 |
| `DELETE` | `/sessions/{id}` | 删除会话 |

### 聊天
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 流式聊天（SSE），支持 `use_rag` 切换 RAG 模式 |

### 文档管理
| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/documents/upload` | 上传文档（multipart/form-data） |
| `GET` | `/api/documents` | 列出所有文档 |
| `DELETE` | `/api/documents/{object_key:path}` | 删除文档（三步顺序：BM25→持久层→元数据） |

---

## 核心数据流

### 文档摄入（写入链路）

```
用户上传文件 → MinIO 存储原始文件
    → ParserCoordinator 解析为 Markdown
    → SemanticSplitter 语义分块
    → ParentChildBuilder 生成父子块
    → EmbeddingService 向量化（BGE → 1024 维）
    → MilvusVectorStore 存入父子双集合
    → PostgreSQL 记录文档元数据
    → BM25 内存索引增量更新
```

### 知识检索（读取链路）

```
用户提问（use_rag=true）
    → BM25 关键词召回 + Milvus 向量语义召回（并行）
    → RRF 融合排序（k=60）
    → （可选）bge-reranker-large 重排序 → top-N
    → 拼接 context → LLM 生成增强回答
    → SSE 流式返回 + 来源文档标注
```

### Agent 执行流程

```
START
  → load_context（加载 PG 聊天历史 + 文件记忆）
  → intent_route（LLM 判断意图，二选一分流）
      ├─ general_chat → result_synthesis → END
      └─ use_tool → tool_execute（执行工具）→ result_synthesis → END
```

防死循环：`max_turns` 硬限制 + token 预算上限双重保障。

---

## 开发路线图

```
✅ 阶段 1-4：Web 聊天 + 持久记忆
             FastAPI + SQLite → PostgreSQL 迁移完成
✅ 阶段 5：RAG 知识库
             混合检索 + 语义分块 + 父子块 + 重排 + 评估，全部完成
✅ 阶段 6.1：Agent 框架骨架
             LangGraph 4 节点 + 工具注册表 + BOUND 约束，端到端验证通过
🔄 阶段 6.2：工具扩展 + 记忆系统
             网页搜索 + MEMORY.md 自动维护（进行中）
⬜ 阶段 6.3：自我检查闭环 + 双角色分工
⬜ 阶段 6.4+：多智能体 + 工作流编排
⬜ 阶段 7：多平台接入（企微/微信/Telegram）
```

---

## 开发指南

### 启动前必读（按顺序）

1. **`docs/上次会话成果.md`** — 最新进度、代码状态、下一步任务（唯一每次更新的文件）
2. **`docs/阶段5技术方案.md`** — RAG 完整架构（按需）
3. **`docs/阶段6技术方案.md`** — Agent 编排架构（按需）

### 协作模式：Mode E（骨架式教学开发）

本项目采用 5 模式渐进式协作（A-B-C-D-E），默认使用 **Mode E**：

| 模式 | 适用场景 | 谁写代码 |
|------|---------|---------|
| **E（默认）** | 新模块/新文件开发 | Claude 生成骨架 → 逐函数教学 → 用户实现 |
| A | 用户明确要求 / 全新语言领域 | Claude 写完整代码 |
| C | Bug 修复 | Claude 提议修改，用户实现 |
| D | 参考已有代码的新功能 | Claude 提议，用户实现 |

**Mode E 四步流程**：骨架生成 → 逐函数教学（知识点+示例+任务） → 互动问答 → 收尾记录。

Python 后端用户自己写，JS 前端 Claude 写。详见根目录 `CLAUDE.md` 中的完整说明。

### 集成快照（app.py 里程碑备份）

每次在 `app.py` 上完成一个功能集成、验证通过后，把当时的入口文件存一份快照。项目根目录始终保留**最新可运行版**，历史版本在 `backups/snapshots/` 里按里程碑归档。

**只备份 `app.py`**，不备份 agent/、前端、RAG 等（那些用 Git 管理，避免占磁盘）。

**工作流**：

```
改 app.py → 本地验证（uv run app.py）→ 集成完成 → 创建快照 → 继续下一功能
```

**创建快照**：

```bash
cd personal_assistant
uv run python scripts/snapshot.py <阶段标签> "<本次集成说明>"
```

示例：

```bash
uv run python scripts/snapshot.py stage62_web_search "P2 web_search 接入"
# → backups/snapshots/20260621_stage62_web_search/app.py
```

| 参数 | 说明 |
|------|------|
| `阶段标签` | 英文+下划线，如 `stage62_web_search`（脚本自动加日期前缀） |
| `"说明"` | 可选，一句话描述本次集成了什么 |

**查看已有快照**：

```bash
cat backups/manifest.json
ls backups/snapshots/
```

**恢复某个版本**：

```bash
cp backups/snapshots/20260621_stage62_agent_graph/app.py app.py
uv run app.py   # 恢复后重启服务
```

**建议习惯**：重要集成 = 先 snapshot，再 git commit（快照记里程碑，Git 记日常 diff）。

更多细节见 [`backups/README.md`](backups/README.md)。

---

## 用户档案

- **GitHub**：[Yukang317](https://github.com/Yukang317)
- **技术方向**：Python 后端 / LLM 应用开发
- **已掌握**：Prompt 工程、FastAPI、LangGraph、MCP、RAG（混合检索+重排+评估）、PEFT/LoRA
- **做过的主要项目**：SAGT 销冠智能体、LlamaIndex RAG（混合检索+语义分块）、MCP 多智能体系统
- **学习风格**：逐行拆解、追根究底、笔记详细、擅长阅读和理解代码

### 已学项目参考索引

需要参考某个能力的实现时，去看对应的学习项目：

| 需要的能力 | 参考项目位置 |
|-----------|-------------|
| FastAPI + JWT + SSE | `学习过程与学习过程中的项目/AITest1_copy/大模型基础/案例/` |
| 混合检索+语义分块+重排+评估 | `学习过程与学习过程中的项目/AITest1_copy/Agent智能体开发/7-llamaindex项目/` |
| LangGraph 多智能体 + MCP | `学习过程与学习过程中的项目/AITest1_copy/Agent智能体开发/mcp-project/` |
| Milvus + MinIO + 父子块 | `学习过程与学习过程中的项目/AITest1_copy/私有化微调/企业知识库项目/` |
| 文档解析器策略模式 | `学习过程与学习过程中的项目/mildoc/mildoc_index/parser/` |

---

## 相关资源

- [DeepSeek API 文档](https://platform.deepseek.com/api-docs/)
- [Milvus 向量数据库文档](https://milvus.io/docs/)
- [LangGraph 编排框架](https://langchain-ai.github.io/langgraph/)
- [MinIO 对象存储](https://min.io/docs/)
- [RAGAS 评估框架](https://docs.ragas.io/)
- [BGE Embedding 模型](https://huggingface.co/BAAI/bge-large-zh-v1.5)

---

**作者**：[Yukang317](https://github.com/Yukang317)
**最后更新**：2026-06-21
