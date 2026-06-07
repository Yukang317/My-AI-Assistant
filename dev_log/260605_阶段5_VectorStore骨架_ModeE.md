# 阶段 5 — rag/storage/vector_store.py（Mode E 骨架生成 + __init__ 教学）

**日期**：2026-06-05（第 7 次会话）
**模式**：Mode E（骨架式教学开发）
**内容**：重写 vector_store.py 完整 14 方法骨架，讲解概念层次，用户实现 __init__

---

## 〇、做了什么 & 为什么

**当前位置**：阶段 5 RAG，路线 1（文档摄入）。Step 1-3（config / embedding / minio_client）已完成，这次做 Step 4 — Milvus 向量存储封装。

**本次完成了什么**：
1. 用 Mode E 生成完整骨架（14 个方法，全部签名 + docstring + TODO）
2. 讲解了 IVF_FLAT / BM25 / 向量检索 / RRF 融合 / 父子分块的层次关系
3. 用户实现了 `__init__` 方法（MilvusClient 连接 + 参数默认值）
4. 用户新增了 `Config.MILVUS_DB_NAME` 配置项

**这个模块在架构中的作用**：向量存储是路线 1（摄入）和路线 2（检索）的交叉点。路线 1 往里写向量，路线 2 从里面搜向量。Milvus 在这里相当于"搜索引擎"，比 PostgreSQL 的 LIKE 查询强在能按语义相似度搜。

---

## 一、代码全貌

### Config 新增项

```python
# rag/config.py — 新增 MILVUS_DB_NAME
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME", "personal_assistant")
```

### __init__ 实现（用户手写）

```python
def __init__(self, host=None, port=None, db_name=None):
    host = host or Config.MILVUS_HOST
    port = port or Config.MILVUS_PORT
    db_name = db_name or Config.MILVUS_DB_NAME

    try:
        # pymilvus 用 uri 参数，不是分开的 host/port
        self.client = MilvusClient(uri=f"http://{host}:{port}", db_name=db_name)
        self.client._init_collections()  # ⚠️ 应为 self._init_collections()
    except Exception as e:
        raise ConnectionError(f"无法连接到 Milvus: {e}")
```

### 骨架全景（14 方法总览）

| # | 方法 | 可见性 | 职责 |
|---|------|--------|------|
| 1 | `__init__` | 公开 | 连接 Milvus → 初始化 Collection |
| 2 | `_build_parent_schema` | 私有 | 构建 parent_chunks 的 9 字段 Schema |
| 3 | `_build_child_schema` | 私有 | 构建 child_chunks 的 9 字段 Schema |
| 4 | `_init_collections` | 私有 | 幂等创建两个 Collection + 索引 |
| 5 | `_create_collection_if_not_exists` | 私有 | 单 Collection 创建/加载逻辑 |
| 6 | `_create_index_if_not_exists` | 私有 | IVF_FLAT + IP 索引创建 |
| 7 | `insert_parent_child` | 公开 | 批量写入父子向量 |
| 8 | `search` | 公开 | 子集合检索 → 父集合回溯完整上下文 |
| 9 | `delete_by_doc_path` | 公开 | 按 MinIO object_key 删除全部向量 |
| 10 | `check_document_exists` | 公开 | 去重判断：文档是否已索引 |
| 11 | `get_indexed_doc_md5s` | 公开 | 获取全量 MD5 映射（增量索引用） |
| 12 | `flush` | 公开 | 强制刷盘 |
| 13 | `load_collection` | 公开 | 加载到内存（搜索前） |
| 14 | `release_collection` | 公开 | 释放内存（省 ~200MB） |

---

## 二、自提疑问 & 解答

### Q1：IVF_FLAT 已经建了索引，为什么还需要 BM25、向量检索、RRF 融合、父子分块？这不是重复了吗？

**背景**：用户看到技术方案里有 IVF_FLAT 索引 + BM25 + 向量 + RRF + 父子分块，觉得都是"索引/检索相关"，以为重复了。

**解答**：完全不是重复，分属两个不同层面：

| 层次 | 做的事情 | 类比 |
|------|---------|------|
| **父子分块** | 文档怎么切成小块和大块 | 菜怎么切（切块 vs 切丝） |
| **BM25 + 向量** | 用不同方式找相关文档 | 按菜名找 vs 按口味找 |
| **RRF 融合** | 合并两种搜索结果的排名 | 两份菜单取加权综合排名 |
| **IVF_FLAT** | 让向量搜索跑得更快 | 图书馆的书分类上架（加速查找） |

IVF_FLAT 只管一件事：100 万个向量 → 分 1024 个簇 → 搜索时只搜最近的 64 个簇。这只是**加速技巧**，不改变搜索结果本身。BM25 和向量是两种完全不同的"找东西"方式，各有优势，需要 RRF 合并排名。

**关键认知**：IVF_FLAT 在 Milvus 内部，用户感知不到。BM25/RRF/父子分块在应用层，是我们要写的代码。

### Q2：MilvusClient 连接后，创建/加载 Collection 和客户端是什么关系？

**解答**：客户端是"电话线"，Collection 是电话那头要操作的"柜子"。

```
self.client ──HTTP──→ Milvus 服务
                        ├── db: personal_assistant
                        │   ├── parent_chunks  ← 一个 Collection（类似 PG 的表）
                        │   └── child_chunks   ← 另一个 Collection
                        └── db: mildoc（其他项目）
```

类比：`psycopg2.connect()` 建立数据库连接 → 所有 SQL 都通过它发过去。同样，Collection 的创建/加载/搜索都是 `self.client.xxx()` 调用。

---

## 三、踩坑记录

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `self.client._init_collections()` — `_init_collections` 是 `MilvusVectorStore` 自己的方法，不是 `MilvusClient` 的方法，调用会报 `AttributeError` | 把类自己的方法和客户端 API 混了 | 应改为 `self._init_collections()`（不带 `client.`） |
| 2 | 最初用 `MilvusClient(host=..., port=...)` — pymilvus 不接受分立的 host/port | pymilvus API 用 `uri` 参数 | 改为 `MilvusClient(uri=f"http://{host}:{port}")` ✅ 已修复 |
| 3 | 最初用 `Config.MILVUS_DB_NAME` 但 Config 里没有这个属性 | 新增配置项 | 在 config.py 新增了 `MILVUS_DB_NAME` ✅ 已修复 |

---

## 四、协作模式评估

本次用 **Mode E（骨架式教学开发）** — Step 1 骨架生成 + Step 2 第一个函数教学。

**好的地方**：
- 用户提出了很好的概念性问题（IVF_FLAT vs 其他检索机制的关系），说明在真正思考架构
- `__init__` 实现基本正确，理解了参数默认值模式和 uri 写法

**需要调整**：
- 本次只推进了 1/14 个函数（__init__），速度较慢。后续可以尝试一次教 2 个简单函数
- 用户时间有限（"百忙之中偷闲"），每次会话短，需要控制单次教学量

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/config.py` | ✅ 小幅修改 | 新增 `MILVUS_DB_NAME` 配置 |
| `rag/storage/vector_store.py` | 🔄 进行中 | 骨架已生成（14 方法），`__init__` 已实现（有 1 个小 bug），剩余 13 个方法待教学 |

---

## 六、下一步

继续 Step 4 剩余 13 个函数的 Mode E 教学，下一个函数：**`_build_parent_schema`**（构建 Milvus Schema）。

注意事项：
- 先修 `__init__` 的 bug：`self.client._init_collections()` → `self._init_collections()`
- `_build_parent_schema` 和 `_build_child_schema` 结构相似，可以一次教两个
- 当前 Milvus 未启动，schema 构建函数可以先写代码但不能跑测试
