# 阶段 5 — VectorStore 完整审查与修复

**日期**：2026-06-06
**模式**：C/D（用户实现 → Claude review → 用户修复 → 反复迭代）
**内容**：完成 vector_store.py 全部 14 个方法实现，经过 3 轮代码审查修复 5 个 bug，讨论 child schema 是否需要 doc_type

---

## 〇、做了什么 & 为什么

上次会话（6/5）把 vector_store.py 的骨架和 8 个核心方法完成了（`__init__` → `search`）。本次：
1. 完成剩余 6 个方法（`delete_by_doc_path`, `check_document_exists`, `get_indexed_doc_md5s`, `flush`, `load_collection`, `release_collection`）
2. 经过 3 轮代码审查，修复了 5 个运行时 bug
3. 讨论了嵌套在审查过程中的技术决策（child schema 缺字段）

这个文件是 RAG 存储层的核心——所有向量写入和检索都经过它。写完之后整个 Step 4 就完整了。

---

## 一、代码全貌 — 本次新增的 6 个方法

### `delete_by_doc_path(doc_path_name) -> Dict[str, int]`

```python
def delete_by_doc_path(self, doc_path_name: str) -> Dict[str, int]:
    if not doc_path_name or not doc_path_name.strip():
        raise ValueError("doc_path_name 为空或空白")
    
    filter_expr = f'doc_path_name == "{doc_path_name}"'
    parent_result = self.client.delete(
        collection_name=Config.MILVUS_PARENT_COLLECTION, filter=filter_expr
    )
    child_result = self.client.delete(
        collection_name=Config.MILVUS_CHILD_COLLECTION, filter=filter_expr
    )
    # delete() 返回 {"delete_count": N}，和 insert() 返回格式一致
    return {
        "parent_deleted": parent_result["delete_count"],
        "child_deleted": child_result["delete_count"],
    }
```

**关键点**：最初版本是先 delete 再 query 拿计数——但数据已经删了，query 永远返回 0。pymilvus 的 CUD 操作都自带返回计数，不需要二次查询。

### `check_document_exists(doc_path_name) -> bool`

```python
if not doc_path_name:
    raise ValueError("doc_path_name 为空")

filter_expr = f'doc_path_name == "{doc_path_name}"'
results = self.client.query(
    collection_name=Config.MILVUS_CHILD_COLLECTION,
    filter=filter_expr,
    output_fields=["id"],  # 只要 id，减少数据传输
    limit=1,               # 有一条就够
)
return len(results) > 0
```

在 child_collection 查（轻量），只需要知道存不存在。用于 backfill_update 的检查逻辑。

### `get_indexed_doc_md5s() -> Dict[str, str]`

```python
results = self.client.query(
    collection_name=Config.MILVUS_PARENT_COLLECTION,
    output_fields=["doc_path_name", "doc_md5"],
    limit=1000,  # 当前数据量小，1000 条足够
)
return {row["doc_path_name"]: row["doc_md5"] for row in results}
```

从 parent 取所有已索引文档的 MD5。用于增量索引时对比：MD5 相同→跳过，不同→更新。

### 三个生命周期方法

```python
def flush(self) -> None:
    # 合并两个集合一次调用
    self.client.flush([Config.MILVUS_PARENT_COLLECTION, Config.MILVUS_CHILD_COLLECTION])

def load_collection(self) -> None:
    self.client.load_collection(Config.MILVUS_PARENT_COLLECTION)
    self.client.load_collection(Config.MILVUS_CHILD_COLLECTION)

def release_collection(self) -> None:
    self.client.release_collection(Config.MILVUS_PARENT_COLLECTION)
    self.client.release_collection(Config.MILVUS_CHILD_COLLECTION)
```

来自 mildoc 的生产实践：
- `flush`：索引后强制刷盘 → 防止 Milvus 重启丢数据
- `load_collection`：搜索前加载到内存 → 已加载时重复调用无害
- `release_collection`：索引用完后释放 → ECS 只有 3.5GB 内存，省 ~200MB

---

## 二、3 轮审查发现的全部 Bug

### 第一轮：原始实现审查

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `search()` output_fields 包含 `doc_type` 但 child schema 没这个字段 | 骨架阶段 child schema 漏定义了 `doc_type` | 补充字段 + 更新 output_fields + 返回字典 |
| 2 | `search()` parent 查询用的 filter 表达式无法被 Milvus 解析 | Python `str(list)` 产生单引号：`['a', 'b']`，Milvus 要求双引号：`["a", "b"]` | `json.dumps(parent_ids)` |
| 3 | `delete_by_doc_path` 先 delete 再 query 计数永远为 0 | Milvus 删除后数据已标记删除，再查同一条件返回空 | 直接取 `delete()` 返回值 `{"delete_count": N}` |
| 4 | `delete_by_doc_path` 返回字符串 `"0 条"` 但类型签名是 `Dict[str, int]` | 旧代码用 f-string 拼接 | 返回 `parent_result["delete_count"]` int |
| 5 | `_create_index_if_not_exists` 有冗余的 `list_indexes()` 调用 | 结果未被使用，`has_index()` 已足够 | 注释掉 |

### 第二轮：用户修复后审查

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 6 | `search()` 返回字典读 `hit["entity"]["doc_type"]` 但 output_fields 换成了 `embedding_model` | 用户把 output_fields 里的 `doc_type` 换成了 `embedding_model`（child 有），但忘记同步改返回字典 | 最终选择了把 `doc_type` 加回 child schema 的方案 |
| 7 | `flush()` 只刷了 parent，漏了 child | 只传了一个集合 | `self.client.flush([col1, col2])` |

### 第三轮：最终修复 ✅

child schema 加了 `doc_type`，output_fields 和返回字典都改回 `doc_type`，docstring 同步更新。flush 两个集合一起刷。全部通过。

---

## 三、自提疑问 & 解答

### Q1：child schema 要不要加 `doc_type`？

**背景**：child 原本只有 8 个字段（没 doc_type 也没 doc_md5），parent 有 9 个。但 `search()` 的 `filter_expr` 参数文档写了 `'doc_type == "md"'`——如果 child 没这个字段，按文件类型过滤搜索就做不了。

**需求角度**：
- 搜索过滤：用户想只搜 md 文件 → 需要 child 有 doc_type（ANN 搜索时下推过滤，不在搜索集合的字段做不了）
- 返回展示：搜索结果告诉用户"这条来自 md 文件"比"embedding_model=BAAI/bge-large-zh-v1.5"有用得多
- 写入成本：indexer 调用 insert 时已经带着 doc_type，零额外开销

**技术角度**：
- 存储成本：VARCHAR(50) × 10000 条 ≈ 0.5MB，可忽略
- 不下推的代价：只能先搜 top_k×N 再后置过滤 → 破坏 top_k 语义，浪费搜索带宽

**结论**：加。不是为了一致性，而是因为需求上需要支持按文件类型过滤搜索。

### Q2：full_update / backfill_update 逻辑放哪里？

用户分享了一段参考代码（MinIO 桶遍历 → 检查是否存在 → 分块→ Embed → 写入 Milvus → flush）。这属于编排层逻辑，应该放在新文件 `rag/storage/indexer.py`，而不是 `vector_store.py` 里。`MilvusVectorStore` 只管低层的 Milvus CRUD。

---

## 四、协作模式评估

本次用了 C/D 混合：用户实现 Claude review，反复迭代。效果很好——每个 bug 都是用户在实现过程中自己踩出来的，修完记忆深刻。三轮审查下来文件越来越干净。

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/storage/vector_store.py` | ✅ | Step 4 全部 14 个方法完成，child schema doc_type 已补充，0 个待修 bug |

---

## 六、下一步

`vector_store.py` 完成。进入 Step 5 `rag/parser/`（文档解析器），Mode E 骨架式教学。
