# 阶段 5 — Step 13 Docker 基础设施

**日期**：2026-06-10
**模式**：手把手教学（用户自己写，Claude 讲解+Review）
**内容**：docker-compose.yml 编写 + etcd 深度讲解

---

## 〇、做了什么 & 为什么

**当前在项目中的位置**：

```
路线1（文档摄入）✅ → 路线2（知识检索）✅ → FastAPI 集成 ✅ → 前端集成 ✅
                                                                        ↓
                                                              Step 13 🔄 Docker 启动
```

RAG 系统的三方存储（MinIO + Milvus + PostgreSQL）都是独立服务，不能直接 `uv run` 启动。用 Docker Compose 编排 4 个容器。

**本次完成**：
- 启动 Docker daemon（`sudo systemctl start docker`）
- 编写 `docker-compose.yml`（4 个容器：etcd / MinIO / Milvus / PostgreSQL）
- 代码 Review 发现 5 个问题并修复
- etcd 深度讲解

**4 个容器间的关系**：

```
etcd(:2379) ←──── Milvus(:19530) ────→ MinIO(:9000)
                  gRPC ↑                    ↑ S3 API
                你的 Python 代码 ──────────┘
                       │
                       └──→ PostgreSQL(:15432)
```

---

## 一、docker-compose.yml 关键设计决策

### 1.1 为什么 etcd/MinIO/Milvus 用 `network_mode: "host"`？

Milvus 内部代码硬编码了 `127.0.0.1:2379`（etcd）和 `127.0.0.1:9000`（MinIO）。默认 bridge 模式下每个容器有自己的独立 localhost，Milvus 容器里的 `127.0.0.1` 指向的是自己而不是宿主机，连不上 etcd/MinIO。

host 模式让 3 个容器共享宿主机的网络栈，`127.0.0.1` 对它们来说都是同一个地址。

### 1.2 为什么 PG 用端口映射而不是 host 模式？

PG 只被宿主机上的 `step23_rag.py` 访问，不需要被其他容器访问。端口映射（`15432:5432`）比 host 模式更干净——如果宿主机自己也有一个 PG 跑在 5432 上，不会冲突。

### 1.3 各容器的关键配置

| 容器 | 关键配置 | 为什么 |
|------|---------|--------|
| etcd | `network_mode: host` | 让 Milvus 能用 127.0.0.1 访问 |
| etcd | 4 个环境变量（compaction/retention/quota/snapshot） | 防止存储无限膨胀 |
| etcd | `--name personal_assistant` | 给 etcd 节点命名 |
| MinIO | `network_mode: host` | 同上 |
| MinIO | `--console-address ":9001"` | 9000=API, 9001=Web 管理台 |
| Milvus | `network_mode: host` | 同上 |
| Milvus | `MINIO_ADDRESS`（不是 `MINIO_ENDPOINTS`） | 环境变量名必须精确匹配 |
| Milvus | `seccomp:unconfined` | Milvus 的 C++ 底层用了被内核限制的系统调用 |
| Milvus | healthcheck 端口 `9091`（不是 19530） | 19530=gRPC 数据面, 9091=HTTP 控制面 |
| Milvus | `start_period: 90s` | Milvus 启动慢，给足初始化时间 |
| Milvus | `depends_on` condition = `service_healthy` | 等 etcd/MinIO 健康检查通过后才启动 |
| PG | `POSTGRES_DB: personal_assistant` | 容器首次启动自动创建库 |
| PG | `POSTGRES_PASSWORD: ""` | 本地开发不需要密码 |
| PG | Alpine 镜像（仅 ~250MB） | 比标准版小将近一半 |

---

## 二、自提疑问 & 解答

### Q1：etcd 到底是什么？为什么 Milvus Standalone（单机）也需要它？

**背景**：你说"讲解一下 etcd 的知识点"。

**解答**：

etcd 是一个强一致性的分布式 key-value 存储，用 Raft 共识算法保证多节点对"当前状态"达成一致。

**核心特性：多版本并发控制（MVCC）**。每次写入不覆盖旧值，而是追加一个新版本（revision）。这让 etcd 支持 watch 机制——客户端可以从某个 revision 开始持续监听变更。

**为什么 Standalone 也需要 etcd**：Milvus Standalone 只是把分布式架构的所有组件（data node、query node、index node、proxy）压缩进一个进程运行。内部代码和分布式版本是同一套——它仍然假装自己是"集群中唯一的一个节点"，往 etcd 里写节点信息、segment 分布、collection schema。

类比：一个人吃饭时仍然把菜盛在碗里用筷子夹——"一个人吃"不意味着"不需要餐具"。

### Q2：etcd 的 4 个环境变量是什么原理？

**解答**：

4 个变量控制 etcd 的"呼吸系统"——写→膨胀→压缩→快照→回收：

| 变量 | 作用 | 生活类比 |
|------|------|---------|
| `AUTO_COMPACTION_MODE=revision` | 按修订号触发清理（不是按时间） | "每写满 1000 页就清理"而不是"每周清理一次" |
| `AUTO_COMPACTION_RETENTION=1000` | 每个 key 保留最近 1000 个版本 | 保留最近 1000 页，更早的撕掉 |
| `QUOTA_BACKEND_BYTES=4294967296` | 存储上限 4GB，超了拒绝写入 | 笔记本最多 4GB，写满了报警 |
| `SNAPSHOT_COUNT=50000` | 每 5 万次写做一次磁盘快照 | 每写 5 万字翻一页新纸，之前的归档 |

**它们的关系**：
```
写操作 → WAL 日志增长
       → 达到 50000 次 → 快照（SNAPSHOT），清理旧 WAL
       → key 版本超过 1000 → 压缩（COMPACTION），删老旧版本
       → 总数据超过 4GB → 拒绝写入（QUOTA），强制查原因
```

### Q3：MINIO_ADDRESS 和 MINIO_ENDPOINTS 有什么区别？

**背景**：Review 时发现你写成了 `MINIO_ENDPOINTS`，应该是 `MINIO_ADDRESS`。

**解答**：

Milvus 社区文档里，环境变量命名在不同版本间有变化。v2.4.0 读的是 `MINIO_ADDRESS`。`MINIO_ENDPOINTS`（带 S，复数）在旧版文档中出现过，但实际 Milvus 代码里不认这个名字。

**排查方法**：去看 Milvus 源码或官方 docker-compose 模板，不看二手博客。环境变量名是精确匹配的，多一个字母都不行。

### Q4：Milvus 为什么有 19530 和 9091 两个端口？

**解答**：数据面/控制面分离是基础设施软件的常见模式。

| 端口 | 协议 | 用途 | 类比 |
|------|------|------|------|
| 19530 | gRPC | 数据面：插向量、搜向量（你的 Python 代码用） | 餐厅前厅：服务员端菜给客人 |
| 9091 | HTTP | 控制面：健康检查、metrics、状态查询（运维用）| 餐厅后厨：备菜、盘点、打扫 |

你的 `step23_rag.py` 连 `19530` 做向量检索，Docker healthcheck 连 `9091` 检查服务是否存活。

---

## 三、踩坑记录

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | etcd/minio 缺 `network_mode: "host"` | 用户写了 milvus 的 host 模式但忘了 etcd 和 minio | 三个容器都加 host 模式 |
| 2 | `MINIO_ENDPOINTS` 拼写错误 | Milvus v2.4.0 读的是 `MINIO_ADDRESS` | 改为 `MINIO_ADDRESS` |
| 3 | `depends_on` 用 `service_started` | 注释写对了但代码写错了，容器起来≠服务就绪 | 改为 `service_healthy` |
| 4 | Milvus healthcheck 端口写 19530 | 19530 是 gRPC 端口，健康检查 HTTP 端点不在那里 | 改为 9091，路径 /healthz |
| 5 | Milvus healthcheck 缺 `start_period` | Milvus 启动慢（30-60s），默认 healthcheck 立刻开始会误判不健康 | 加 `start_period: 90s` |

---

## 四、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `docker-compose.yml` | ✅ | 用户亲手编写，经 Review 修复 5 个问题后完成 |

---

## 五、下一步

**第 3 步**：`docker compose up -d` 启动所有容器
**第 4 步**：验证 PostgreSQL `personal_assistant` 数据库
**第 5 步**：`uv run step23_rag.py` → 浏览器测试

**注意**：启动前确认 `./data/` 目录存在，compose 会自动创建；确保宿主机端口 2379/9000/9001/19530/15432 未被占用。
