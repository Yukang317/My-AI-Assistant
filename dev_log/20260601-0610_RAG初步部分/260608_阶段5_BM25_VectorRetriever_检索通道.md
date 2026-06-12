# 阶段5 BM25 + VectorRetriever — 检索通道编码完成

**日期**：2026-06-08
**文件**：`rag/retrieval/bm25.py` + `rag/retrieval/vector_retriever.py`
**上下文**：Mode E 骨架式教学，路线2（知识检索）第一批文件

---

## BM25Index — 8 个函数全部完成

### 架构角色

```
路线1（文档摄入）产出 chunks         路线2（知识检索）消费
════════════════════════            ════════════════════

indexer.py                          bm25.py
_prepare_chunks_for_milvus()        build_index(chunks, doc_ids)
    │                                    │
    │ 子块文本 + doc_ids                  │ 分词 + 建 BM25Okapi
    └────────────────────────────────────→│
                                          │
                                          ├─→ search("DeepSeek怎么调")
                                          │   返回: [{chunk_index, doc_id, score, text}]
                                          │
                                          ├─→ add_chunks(新chunks)  ← 用户新上传文件
                                          ├─→ remove_by_doc_ids(ids) ← 用户删除文件
                                          ├─→ is_built()
                                          └─→ get_stats()
```

### 核心数据流（search 调用链）

```
用户问："DeepSeek 怎么调？"
        │
        ▼
_tokenize("DeepSeek 怎么调？")
        │
        ▼
["DeepSeek", "怎么", "调"]          ← 查询分词
        │
        ▼
_bm25_model.get_scores(["DeepSeek", "怎么", "调"])
        │
        │  BM25Okapi 遍历全部 chunk，
        │  用 IDF × 词频 算出每个 chunk 的分数
        │
        ▼
[1.2, 3.8, 0.5, 2.1, ...]         ← 每个 chunk 一个分数
        │
        ▼
np.argpartition + np.argsort       ← 取 top_k 最高分
        │                          ┌──────────────────────────┐
        ▼                          │  corpus 和 doc_ids 介入   │
                                   │                          │
chunk_index ──→ self.corpus[i]  ──→ "DeepSeek API 的 base_url..."
        ──→ self.doc_ids[i]  ──→ "documents/deepseek笔记.md"
                                   └──────────────────────────┘
```

### 三个属性的角色（一一对应并行数组）

| 属性 | 存什么 | search 时的作用 |
|------|--------|----------------|
| `_bm25_model` | IDF 表 + 词频统计（统计模型） | `chunk_index → 分数`（只有分数，没有原文） |
| `corpus` | 原始 chunk 文本列表 | `chunk_index → text`（返回文本内容） |
| `doc_ids` | 文档标识列表 | `chunk_index → 文件路径`（标注来源） |

**关键认知**：`_bm25_model` 不是文档、不是文本，而是一个统计模型——它知道"每个词在哪些文档里、出现了几次、这个词有多稀有"。

### 已覆盖知识点

| 函数 | 关键知识点 |
|------|-----------|
| `__init__` | rank_bm25 只负责算法不管分词；corpus 和 doc_ids 是并行数组；`_bm25_model=None` 作为"未构建"标记 |
| `_tokenize` | jieba 精确模式；过滤空格 token（无语义价值、污染 IDF）；英文模式用 `str.split()` 即可 |
| `build_index` | BM25Okapi 接受 `List[List[str]]`（已分词）；fail-fast 先验证再分词；三个操作必须原子完成 |
| `search` | `np.argpartition` 部分排序 O(n) → `np.argsort` 对 top_k 排序 O(k log k)；numpy fancy indexing `score[top_indices]`；`[::-1]` 升序转降序；score=0 不过滤（RRF 融合时需要） |
| `add_chunks` | BM25 没有真正的增量——IDF 依赖全量语料库，必须重建；场景：用户第 3 天新上传文件，不用从 DB 重读旧数据 |
| `remove_by_doc_ids` | `set()` 做 O(1) 查找（不用 list 的 O(n)）；删除计数必须在 `build_index` 之前算（防御性编程）；全部 chunk 被删时手动清空三个属性（不能调 `build_index([], [])`） |
| `is_built` | `_bm25_model is not None` 判断（存储对象实例，非布尔值） |
| `get_stats` | `set(doc_ids)` 去重计算唯一文件数（一个文件切 20 个 chunk 在 doc_ids 里出现 20 次） |

### BM25 索引完整生命周期

```
build_index（新建）→ search（查询）→ add_chunks（追加）→ search（继续查）
                                    → remove_by_doc_ids（删除）→ search（继续查）
```

---

## VectorRetriever — 5 个函数全部完成

### 架构角色

```
                 ┌── bm25.py ─────→ 关键词匹配结果 ──┐
用户查询 ──→     │                                     ├──→ RRF 融合 → 重排
                 └── vector_retriever.py ──→ 语义匹配结果 ──┘

vector_retriever 内部链路：
  query → _embed_query() → EmbeddingService.embed() → 1024维向量
        → vector_store.search() → Milvus ANN 搜索 + 父块回溯
        → 字段映射（child_id→chunk_id 等） → 统一输出格式
```

### VectorRetriever vs BM25Index 对比

| 维度 | BM25Index | VectorRetriever |
|------|-----------|-----------------|
| 搜索方式 | 关键词精确匹配 | 向量语义相似 |
| 优势 | API 名、术语、代码片段 | "电脑"≈"计算机" 同义表达 |
| 依赖 | 零外部依赖（纯内存） | EmbeddingService + MilvusVectorStore（依赖注入） |
| 索引存储 | 内存（重启丢失） | Milvus（持久化） |
| 增量添加 | 必须全量重建 | 逐条插入（不在此层） |
| 设计模式 | 自管理（自己 new 对象） | 编排层（薄，只做协调） |

### 已覆盖知识点

| 函数 | 关键知识点 |
|------|-----------|
| `__init__` | 依赖注入：构造函数只接收不创建；调用方控制依赖的生命周期；测试时可 mock |
| `_embed_query` | 批量接口适配成单条接口（`embed([q])[0]`）；`strip()` 防御空白查询；fail-fast 在入口 |
| `search` | 编排模式：自己不干活，调别人；字段映射（防腐层）统一 BM25 和 Vector 输出格式，RRF 融合的前提 |
| `search_by_type` | 便利方法 = 参数转换 + 委托；单一真相来源：不重复 search 逻辑 |
| `get_stats` | 可观测性；与 bm25 的 get_stats 互补（配置信息 vs 数据统计） |

### 输出格式统一（BM25 和 Vector 对齐）

| BM25 字段 | Vector 字段 | 含义 |
|-----------|------------|------|
| `chunk_index` | `chunk_id` | chunk 标识（BM25 用索引，Vector 用业务 ID） |
| `doc_id` | `doc_id` | 文档来源路径 |
| `score` | `score` | 相关性分数 |
| `text` | `text`（child_content） | chunk 文本（检索用） |
| — | `parent_text`（parent_content） | 父块完整上下文（Vector 独有） |
| — | `doc_name`, `doc_type` | 文档元信息（Vector 独有） |

---

## 协作模式记录

本次会话采用 Mode E（骨架式教学开发）：
1. Claude 生成完整骨架（所有 import + 函数签名 + docstring + TODO）
2. 逐函数教学：知识点讲解 → 代码提示 → 用户实现 → Claude review
3. BM25 教学过程中用户反馈"讲太专业"，后续调整为项目场景类比讲解
4. 用户独立发现并修复：解包错误、列表 vs 字典遍历、便利方法的业务场景理解

### 用户发现的 Bug（bm25.py）

| # | 函数 | Bug | 修复 |
|---|------|-----|------|
| 1 | remove_by_doc_ids | `filtered_corpus, filtered_doc_ids = [(chunk, did) for ...]` — Python 不能自动解包 tuple 列表 | 手动分拆两个列表 |
| 2 | remove_by_doc_ids | `if not filtered_corpus and filtered_doc_ids:` 永为 False | 改为 `if not filtered_corpus:` |
| 3 | remove_by_doc_ids | 全部删除时调 `build_index([], [])` 会抛 ValueError | 提前 return，手动清空三个属性 |
| 4 | remove_by_doc_ids | `removed_count` 在 `build_index` 之后计算导致永为 0 | 移到 `build_index` 之前计算 |

### 用户发现的 Bug（vector_retriever.py）

| # | 函数 | Bug | 修复 |
|---|------|-----|------|
| 1 | search | `for child_id, parent_id, ... in raw_results` 对字典列表做元组解包 | 改为 `for item in raw_results` + `item["key"]` |
| 2 | get_stats | `self.embedding_service.mode()` 多了括号（是属性不是方法） | 去掉括号 |
| 3 | get_stats | `Config.get_embedding_model()` 方法名错误 | 改为 `Config.get_embedding_model_name()` |

---

## 下一步

`rag/retrieval/rrf_fusion.py` — RRF 融合（BM25 + 向量结果合并排序）
