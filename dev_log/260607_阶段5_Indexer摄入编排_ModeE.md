# 阶段 5 — Indexer 文档摄入编排器（rag/indexer.py）

**日期**：2026-06-07
**模式**：E（骨架式教学开发）— 逐函数教学模式
**内容**：完成 indexer.py 全部 6 个函数，串联 Parser → Chunker → Embedding → MinIO → Milvus 五大模块

---

## 〇、做了什么 & 为什么

### 在项目中的位置

```
路线1（文档摄入）已完成：
✅ Config → ✅ Embedding → ✅ MinIO → ✅ Milvus → ✅ Parser → ✅ Chunker → ✅ Indexer ← 本次
```

Indexer 是路线1的**顶层编排者**（也是最后一块拼图）。前面 6 个模块都在"造零件"，Indexer 负责把它们串成一条完整的流水线。

### 本次完成了什么

全部 6 个函数：

| 函数 | 类型 | 行数 | 说明 |
|------|------|------|------|
| `_compute_md5` | 模块级 | 1 行 | 计算文件 MD5 |
| `_build_object_key` | 模块级 | 5 行 | 构建 MinIO 对象路径 |
| `__init__` | 方法 | 4 行 | 初始化 4 个子模块 |
| `index_document` ★核心 | 方法 | 80 行 | 完整 8 步摄入管线 |
| `_prepare_chunks_for_milvus` | 方法 | 37 行 | dataclass → dict 防腐层 |
| `delete_document` | 方法 | 12 行 | MinIO + Milvus 双清 |

### 在整体架构中的作用

Indexer 自己不写业务逻辑——解析/分块/向量化/存储都在各自模块里。它只负责：
1. **调度的顺序**：先做什么后做什么
2. **全局判断**：MD5 去重（需要同时看 MinIO 和 Milvus，任何子模块都做不了）
3. **异常的传递**：哪步失败报哪步，方便定位

---

## 一、代码全貌

### 函数 1：`_compute_md5(data: bytes) -> str`

```python
def _compute_md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()
```

一行代码。选 MD5 而不是 SHA-256 是因为：这里的用途是内容去重（不是安全校验），MD5 足够区分不同文件，且速度快约 2 倍。

### 函数 2：`_build_object_key(filename: str) -> str`

```python
def _build_object_key(filename: str) -> str:
    now = datetime.now()
    year = now.year
    month = now.month
    index_path = f"documents/{year}/{month:02d}/{filename}"
    return index_path
```

按年月分目录的好处：MinIO 控制台可层级浏览 + 方便按时间范围批量操作 + 避免单目录文件太多。

关键细节：用 `datetime.now()` 而不是文件创建时间——关心的是"什么时候索引的"。

### 函数 3：`__init__`

```python
def __init__(self) -> None:
    self.parser = DocumentParserCoordinator()
    self.embedder = EmbeddingService()
    self.vector_store = MilvusVectorStore()
    self.minio = MinioClient()
```

四个子模块初始化顺序无关紧要（它们之间没有依赖）。

### 函数 4：`index_document` ★核心

完整的 8 步管道，分三阶段：

**第一阶段：准备工作**
```python
is_update = False  # 追踪文件状态：new vs updated

# 1. 文件类型校验 + MIME 映射
file_type = os.path.splitext(filename)[1].lower()
if file_type not in Config.ALLOWED_EXTENSIONS:
    raise ValueError(f"文件类型 {file_type} 不支持")
content_type = CONTENT_TYPE_MAP.get(file_type, "application/octet-stream")

# 2. MD5 + object_key
file_md5 = _compute_md5(file_data)
object_key = _build_object_key(filename)

# 3. MD5 去重检查（三种情况）
if self.vector_store.check_document_exists(object_key):
    existing_md5s = self.vector_store.get_indexed_doc_md5s()
    if existing_md5s.get(object_key) == file_md5:
        return {"status": "skipped", ...}       # 情况A：内容相同，跳过
    else:
        self.delete_document(object_key)         # 情况B：内容更新，先删旧
        is_update = True
```

**第二阶段：分块 + 向量化**
```python
text = self.parser.parse(file_data, file_type)  # bytes → 纯文本
chunks = split_by_type(text, file_type)          # 纯文本 → 语义块
parents, children = build_parent_child(chunks, filename)  # 语义块 → 父子块

parent_texts = [p.content for p in parents]
child_texts = [c.content for c in children]
parent_vectors = self.embedder.embed(parent_texts)  # 父块文本 → 向量
child_vectors = self.embedder.embed(child_texts)    # 子块文本 → 向量

# fail-fast 校验：N 个文本 → N 个向量
if len(parent_vectors) != len(parent_texts):
    raise ValueError(...)
```

**第三阶段：双存储写入**
```python
self.minio.upload(file_data, object_key, content_type)  # 先 MinIO（失败不污染 Milvus）
parent_dicts, child_dicts = self._prepare_chunks_for_milvus(...)  # dataclass → dict
result = self.vector_store.insert_parent_child(parent_dicts, child_dicts)
self.vector_store.flush()  # 强制刷盘

return {"status": "updated" if is_update else "new", ...}
```

### 函数 5：`_prepare_chunks_for_milvus` — 防腐层

```python
def _prepare_chunks_for_milvus(self, parents, children, parent_vectors,
                                child_vectors, doc_name, doc_path_name,
                                doc_type, doc_md5):
    # 入口校验
    if len(parents) != len(parent_vectors) or len(children) != len(child_vectors):
        raise ValueError("向量数量与 chunk 数量不匹配")

    embedding_model = Config.get_embedding_model_name()

    # 父块：zip(chunk, vector) 一一配对
    parent_dicts = []
    for chunk, vec in zip(parents, parent_vectors):
        parent_dicts.append({
            "doc_name": doc_name, "doc_path_name": doc_path_name,
            "doc_type": doc_type, "doc_md5": doc_md5,
            "embedding_model": embedding_model,
            "parent_id": chunk.parent_id, "content": chunk.content,
            "content_vector": vec,
        })

    # 子块：比父块多一个 child_id
    child_dicts = []
    for chunk, vec in zip(children, child_vectors):
        child_dicts.append({
            # ... 同上公共字段 ...
            "parent_id": chunk.parent_id,
            "child_id": chunk.child_id,  # 子块独有
            "content": chunk.content, "content_vector": vec,
        })

    return parent_dicts, child_dicts
```

上游产出 `ParentChunk`/`ChildChunk` dataclass，下游期望 `Dict[str, Any]`。中间夹一层转换，两边各说各的"方言"，互不污染。

### 函数 6：`delete_document` — 双清

```python
def delete_document(self, object_key: str) -> Dict[str, int]:
    if not object_key:
        raise ValueError("object_key 为空")

    self.minio.delete(object_key)  # 文件不存在时静默忽略
    result = self.vector_store.delete_by_doc_path(object_key)  # 返回 {parent_deleted, child_deleted}

    return {
        "minio_deleted": 1,
        "parent_deleted": result["parent_deleted"],
        "child_deleted": result["child_deleted"],
    }
```

只用了 12 行。不是因为它简单，而是 `MinioClient` 和 `MilvusVectorStore` 封装好了底层细节。**好的封装会让编排者很薄。**

---

## 二、知识要点笔记

### 知识点 1：管道模式（Pipeline Pattern）

每一步的输出是下一步的输入，像流水线。好处：出问题时精确定位到"第几步挂了"，每步抛出的异常就是天然的断点。

### 知识点 2：MD5 去重的三种情况

这是编排者独有的全局判断——需要同时查 Milvus（有没有这个 doc_path）和比较 MD5，Parser/Embedder 任何一个子模块都做不了：

| 情况 | 条件 | 处理 |
|------|------|------|
| 新文件 | Milvus 中没有此 object_key | 直接走完整管线 |
| 内容相同 | 已索引 且 MD5 一致 | 跳过，返回 `"skipped"` |
| 内容更新 | 已索引 但 MD5 不同 | 先删旧数据，再走完整管线 |

### 知识点 3：防腐层（Anti-Corruption Layer）

`_prepare_chunks_for_milvus` 就是防腐层：
- 上游产出 `ParentChunk`/`ChildChunk` dataclass（语义清晰）
- 下游期望 `Dict[str, Any]`（Milvus 只认识这个）
- 中间夹一层转换，两边都不需要为对方妥协自己的数据结构

### 知识点 4：MinIO 在 Milvus 之前写入

不是任意的顺序：如果 MinIO 上传失败，可直接抛异常，不污染 Milvus。如果先写 Milvus 再上传 MinIO 失败，就需要回滚——而 Milvus 不支持事务回滚。

### 知识点 5：zip() 的"拉链"比喻

```python
for chunk, vec in zip(parents, parent_vectors):
    # 3 个 chunk + 3 个向量 → 3 轮循环，每轮 chunk 和 vec 都是配好的
```

就像穿衣服拉拉链——左边一个齿（chunk）对应右边一个齿（vector），一一咬合。

### 知识点 6：契约信任 + 防御校验

- `embed()` 的契约是"你给我几个文本，我还你几个向量"，调用时信任它
- 写入 Milvus 前再校验一遍数量——信任让你不用读 embed 源码，校验让你在出问题时第一时间定位到 embedding 环节的锅
- 这是写管道代码的标配组合

### 知识点 7：幂等删除

`minio.delete()` 对不存在的 object 不会抛异常，直接返回成功。这意味着"删一个不存在的东西"和"删一个存在的东西"对外表现一致——这就是幂等。

---

## 三、踩坑记录

本次 coding 共发现 **6 个 bug**：

| # | Bug 现象 | 根因 | 修复 | 函数 |
|---|---------|------|------|------|
| 1 | `AttributeError: 'MilvusVectorStore' object has no attribute 'get_md5s'` | 方法名叫 `get_indexed_doc_md5s()` 不是 `get_md5s()` | 改为正确方法名 | F4 |
| 2 | 删除操作执行了两遍 | 调了 `self.delete_document()` 又调了 `self.vector_store.delete_by_doc_path()`，而前者内部已调后者 | 注释掉重复调用 | F4 |
| 3 | 文件更新后未走摄入管线 | MD5 不匹配的分支里提前 `return`，跳过了 parse→chunk→embed→store | 去掉了 return，改用 `is_update` 标记，让它 fall through | F4 |
| 4 | 校验条件写反：`not A == B or C == D` | `not` 只作用于 `A == B`，而 `C == D` 单独计算——导致子块匹配时也报错（不该报的时候报了） | 改为 `A != B or C != D` | F5 |
| 5 | `TypeError: unsupported operand type(s) for +: 'NoneType' and 'str'` | `minio.delete()` 没有返回值（返回 None），写了 `None + "字符串"` | 直接写死 `"minio_deleted": 1` | F6 |
| 6 | MinIO 删除失败则 Milvus 也无法清理 | try/except 里 re-raise ValueError，阻止了后续的 Milvus 删除 | 去掉 try/except，让流程继续 | F6 |

**教训**：
- Bug 3 和 Bug 6 是**控制流错误**——不该 return 的时候 return 了，不该抛异常的时候抛了。这类 bug 最隐蔽，代码看起来"逻辑正确"，但实际执行路径不对。
- Bug 4 是经典的**布尔代数错误**——`not` 的优先级和括号缺位。`not A or B` ≠ `not (A or B)`。

---

## 四、协作模式评估

**Mode E** 效果很好。本次 session：
- 用户卡在 `index_document` 时主动喊"写不下去了帮帮我"——这正是 Mode E 设计的互动节奏
- 逐函数教学 + 即时 review 让每个 bug 都在当场发现和修复，没有遗留到下一次
- 用户对 `zip()` 遍历语法不确定时直接问"具体遍历怎么写"——不装懂，这很重要

---

## 五、待清理

`indexer.py` 中还有几处残留：
- 第 71、87、176、306、370 行的 `# TODO(human):` 注释（功能已实现，可删除）
- 第 213-214、378、383、386 行的注释掉的调试代码

不影响运行，下次统一清理。

---

## 六、下一步

**下一步文件**：`rag/retrieval/` — 检索管线（路线2-①）
- 第一个要写的：`retrieval/bm25.py` — 本地 BM25 关键词检索器
- 然后是 `retrieval/vector_retriever.py`、`retrieval/rrf_fusion.py`
- **继续用 Mode E**
