# 阶段 5 — VectorStore Search 函数 + 去重实现

**日期**：2026-06-05
**模式**：E（骨架式教学开发，后半段因用户头晕切 A 辅助解释）
**内容**：完成 `insert_parent_child` 回归和 `search` 函数的完整实现，重点讨论了检索去重策略和分块策略

---

## 〇、做了什么 & 为什么

**项目位置**：阶段 5 Step 4 — `rag/storage/vector_store.py`，前面 6 个方法（`__init__` + Schema + Collection 管理）已完成，这次完成数据写入（`insert_parent_child`）和向量检索（`search`）两个核心方法。

**本次完成**：
- `insert_parent_child`：确认代码逻辑正确（上次会话已讨论清楚），补充 `insert_count` API 返回值理解
- `search`：完整的检索链路 —— 子块语义搜索 → parent_id 收集 → 批量 query 父块 → 去重 → 排序返回

**在整体架构中的作用**：`search` 是路线 2（知识检索）的入口函数，上游是 `rag/retrieval/`（检索管线），下游给 RAG service 提供检索结果。

---

## 一、代码全貌

### 1.1 `insert_parent_child` 最终版本

```python
def insert_parent_child(self, parent_chunks, child_chunks) -> Dict[str, int]:
    if not parent_chunks:
        raise ValueError("parent_chunks 列表为空，无法插入")
    if not child_chunks:
        raise ValueError("child_chunks 列表为空，无法插入")

    parent_result = self.client.insert(
        collection_name=Config.MILVUS_PARENT_COLLECTION, data=parent_chunks
    )
    child_result = self.client.insert(
        collection_name=Config.MILVUS_CHILD_COLLECTION, data=child_chunks
    )

    return {
        "parent_count": parent_result["insert_count"],
        "child_count": child_result["insert_count"],
    }
```

**关键点**：`MilvusClient.insert()` 返回 `{"insert_count": N, "ids": [...]}`，`insert_count` 是 pymilvus 写死的返回格式，不是我们定义的。

### 1.2 `search` 完整实现（含去重）

```python
def search(self, query_vector, top_k=10, filter_expr=None) -> List[Dict[str, Any]]:
    # 搜索参数：nprobe 控制搜多少个簇（不是 nlist，nlist 在建索引时就定了）
    search_params = {"nprobe": Config.MILVUS_SEARCH_NPROBE}

    # 默认过滤：只搜当前 embedding 模型的向量（防止新旧模型混搜）
    expr = f'embedding_model == "{Config.get_embedding_model_name()}"'
    if filter_expr:
        expr += f" and {filter_expr}"

    results = self.client.search(
        collection_name=Config.MILVUS_CHILD_COLLECTION,
        data=[query_vector],
        anns_field="content_vector",  # 在哪个向量字段上做 ANN 搜索
        limit=top_k,
        filter=expr,
        search_params=search_params,
        output_fields=["child_id", "parent_id", "content", "doc_name",
                       "doc_path_name", "doc_type"],
    )

    # ⚠️ 注意：results 是 list of lists，results[0] 才是第一条查询向量的命中
    if not results[0]:
        return []

    # Step A：收集所有命中的 parent_id → 去重 → 一次性 query 父块
    parent_ids = list(set(hit["entity"]["parent_id"] for hit in results[0]))

    parents = self.client.query(
        collection_name=Config.MILVUS_PARENT_COLLECTION,
        filter=f'parent_id in {parent_ids}',  # in 表达式一次搞定，不用 N 次网络请求
        output_fields=["parent_id", "content"],
    )
    parent_content_map = {p["parent_id"]: p["content"] for p in parents}

    # Step B：组装 + 去重（同一个 parent_id 只保留 score 最高那条）
    dedup = {}
    for hit in results[0]:
        pid = hit["entity"]["parent_id"]
        score = hit["distance"]
        if pid not in dedup or score > dedup[pid]["score"]:
            dedup[pid] = {
                "child_id": hit["entity"]["child_id"],
                "parent_id": pid,
                "child_content": hit["entity"]["content"],
                "parent_content": parent_content_map.get(pid, ""),
                "doc_name": hit["entity"]["doc_name"],
                "doc_path_name": hit["entity"]["doc_path_name"],
                "doc_type": hit["entity"]["doc_type"],
                "score": score,
            }

    output = list(dedup.values())
    output.sort(key=lambda x: x["score"], reverse=True)
    return output
```

**设计决策**：
- 去重用 dict（O(n)），不用双重循环（O(n²)）
- 父块查询用 `in` 表达式（1 次网络往返），不循环 query（N 次）
- `parent_content_map` 是 `{parent_id: content}` 字典，O(1) 查找

---

## 二、自提疑问 & 解答

### Q1：`search_params` 为什么只有 `nprobe`，没有 `nlist`？

**解答**：nlist 和 nprobe 作用在不同阶段。

| 参数 | 阶段 | 含义 |
|------|------|------|
| `nlist=1024` | 建索引时（`create_index`） | 把全部向量分成 1024 个簇 |
| `nprobe=64` | 搜索时（`search`） | 搜这 1024 个簇中的 64 个 |

nlist 在 `_create_index_if_not_exists()` 里已写死，search 只需要 nprobe。nprobe 越大搜得越准但越慢，64/1024=6.25%，这就是 IVF_FLAT "近似"搜索的本质。

### Q2：`anns_field="content_vector"` 是什么意思？

**解答**：Milvus 一个 Collection 可以存多个向量字段（如 `content_vector` 和 `title_vector`），搜索时必须指定"在哪个向量字段上做相似度匹配"。我们只有一个 `content_vector`，固定写死即可。字段名必须和 Schema 定义一致。

### Q3：搜索到的子块返回父块，会不会返回大量无关内容？

**用户提出的方案**：如果子块覆盖了父块的 80%/90% 以上，返回父块；否则保留子块。

**客观分析**：
- ✅ 担忧的方向是对的：确实可能 1 个子块命中，却返回整个 2048 token 父块（87.5% 无关）
- ❌ 但覆盖率阈值方案有问题：
  1. 无法得知一个父块有几个子块（需要额外记 `total_children` 字段，改 Schema）
  2. 父块的"周边文本"天然相关，子块命中意味着相邻内容大概率也相关
  3. 现代 LLM 对 80% 相关 + 20% 无关容忍度很高
- ✅ 行业标准做法：**去重**（多个子块命中同一父块 → 合并为一条）+ **Reranker**（重排序过滤无关内容）

**结论**：先实现去重，Reranker 在 Step 7 集成。

### Q4：分块是不是直接固定 2048？不同文件类型怎么适配？

**解答**：方案里是两步流水线，不是固定硬切：
1. 语义分块（`SentenceSplitter(separator="\n\n")`，优先在段落边界切）
2. 父子块构建（parent=2048/128，child=512/64）

不同文件类型通过 **Parser 层**（策略模式）统一消化：PDF→PDFParser→纯文本，DOCX→DocxParser→纯文本。Chunker 只吃纯文本，不感知源格式。

**PDF 隐忧**：PDF 解析后丢失标题结构，MarkdownNodeParser 无法利用 H1-H6 层级，分块质量可能不如 Markdown。后续 Step 5-6 需验证，必要时考虑 pymupdf4llm。已记录到 Memory。

---

## 三、踩坑记录

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `if not results` 不会触发空命中保护 | `results` 是 list of lists，无命中时是 `[[]]`（外层非空，内层空），`not results` 为 False | 改为 `if not results[0]` |
| 2 | Child Collection Schema 缺少 `doc_type` 字段 | 骨架生成时漏掉了，`_build_child_schema()` 只定义了 8 个字段，没有 `doc_type` | 🔴 待修复（下次一起改） |
| 3 | `search()` 的 `output_fields` 里请求了 `doc_type`，但 child collection 没有此字段 | 和 Bug 2 是同一根因 | 🔴 待修复 |

---

## 四、协作模式评估

**Mode E 执行情况**：前半段严格按 Step 2（逐函数教学）格式走，用户对 `search` 的嵌套结构感到困惑时，切换为纯解释模式（更接近 A），效果更好。用户头晕时也能跟得上。

**本次经验**：Mode E 的教学步骤（知识点→代码提示→任务）在简单函数上效果很好，但在 `search` 这种多步骤、多层嵌套的函数上，需要先花更多时间讲清数据结构（`results` 长什么样），否则用户会卡在"不知道遍历什么"上。

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/storage/vector_store.py` | 🔄 | 8/14 方法完成：__init__, _build_parent_schema, _build_child_schema, _init_collections, _create_collection_if_not_exists, _create_index_if_not_exists, insert_parent_child, search |
| `rag/config.py` | ✅ | 无需改动 |
| `阶段5技术方案.md` | ✅ v4 | 已参考，无修改 |

**剩余 6 个方法**：delete_by_doc_path, check_document_exists, get_indexed_doc_md5s, flush, load_collection, release_collection

---

## 六、下一步

**下一函数**：`delete_by_doc_path` — 文档删除（含安全检查，逻辑较简单）
**模式**：继续 Mode E
**待修复**：child Schema 缺 `doc_type` 字段（下次开始前先修）
