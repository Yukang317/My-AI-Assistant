# 阶段 5 — Step23 FastAPI 集成完成（修改 5/7 ~ 7/7）

**日期**：2026-06-10
**模式**：C 模式（修改 5）+ A 模式（修改 6-7，用户疲劳切换）
**内容**：完成 step23_rag.py 剩余 3/7 修改——3 个文档端点 + /chat 端点 RAG 分流 + 启动确认

---

## 〇、做了什么 & 为什么

前次会话完成了修改 1-4/7（RAG import + 数据模型 + documents 表 + 懒加载），这次把剩下的 3 个修改全部做完。

**step23_rag.py 在整体架构中的位置**：
```
用户浏览器 → static/index.html
              → POST /chat            (聊天，含 RAG 分流)
              → POST /api/documents/upload  (文档上传)
              → GET  /api/documents   (文档列表)
              → DELETE /api/documents/{key} (文档删除)
              → get_rag_service()     (懒加载，首次 RAG 请求时初始化)
                → RagService          (检索+生成编排)
                → DocumentIndexer     (文档摄入编排)
```

---

## 一、代码全貌

### 1.1 修改 5/7：三个文档端点

#### `POST /api/documents/upload`

```python
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    file_data = await file.read()                        # 读上传文件的全部字节

    indexer = get_indexer()                              # 懒加载 DocumentIndexer
    result = indexer.index_document(file_data, file.filename)  # 8 步摄入管线
    if result["status"] == "skipped":                    # MD5 去重：相同文件跳过
        return result

    doc_id = save_document(...)                           # 写 SQLite 元数据

    if _rag_service is not None:                          # BM25 增量更新
        vs = _rag_service.vector_retriever.vector_store   # 不 new MilvusVectorStore！
        child_chunks = vs.client.query(...)
        update_bm25_after_uploads(...)                    # → add_chunks 全量重建 IDF

    return { "id": doc_id, ... }                          # 含 DB 自增 ID
```

#### `DELETE /api/documents/{object_key:path}`

```python
@app.delete("/api/documents/{object_key:path}")           # :path 让 object_key 里的 / 不被截断
async def delete_document(object_key: str):
    # ① BM25 先删（在 Milvus 删之前查出被删文档的 child_id）
    if _rag_service is not None:
        vs = _rag_service.vector_retriever.vector_store    # 复用，不 new
        child_chunks = vs.client.query(filter=f'doc_path_name == "{object_key}"', ...)
        _rag_service.bm25_index.remove_by_doc_ids(...)     # 过滤 + 重建，比查 Milvus 全量重建更高效

    # ② MinIO + Milvus 删除（持久数据）
    indexer = get_indexer()
    delete_result = indexer.delete_document(object_key)

    # ③ SQLite 删元数据
    db_deleted = delete_document_by_key(object_key)
    return {"ok": True, ...}
```

**三步顺序的设计原因见 Q1**。

### 1.2 修改 6/7：/chat 端点 RAG 分流

#### 新增：`generate_rag_stream_response()`

位置：紧接 `generate_stream_response` 之后。和旧函数成对存在——旧函数直调 DeepSeek，新函数走 RagService。

```python
async def generate_rag_stream_response(session_id: str, question: str):
    rag_service = get_rag_service()
    accumulated = ""

    # query_stream 返回 AsyncGenerator，每次 yield 一个 dict
    async for event in rag_service.query_stream(question):
        if event["type"] == "sources":
            yield SSE: {"type": "sources", "sources": [...], "finished": False}
        elif event["type"] == "delta":
            accumulated += event["content"]
            yield SSE: {"type": "delta", "content": "...", "finished": False}
        elif event["type"] == "complete":
            save_message(session_id, "assistant", accumulated)
            yield SSE: {"type": "complete", "finished": True, ...}
        elif event["type"] == "error":
            raise Exception(...)
```

SSE 事件类型演进：
```
旧版（直调 DeepSeek）：     delta×N → complete（只有文本）
新版（RAG）：sources → delta×N → complete（先推来源，再推文本）
```

#### 修改：`/chat` 端点

在保存用户消息后插入 `if req.use_rag:` 分支，用 `return` 提前退出：

```
POST /chat
├── use_rag=True,  stream=False → get_rag_service().query() → RagChatResponse
├── use_rag=True,  stream=True  → generate_rag_stream_response() → SSE
├── use_rag=False, stream=False → 直调 DeepSeek → ChatResponse（原有逻辑）
└── use_rag=False, stream=True  → generate_stream_response() → SSE（原有逻辑）
```

为什么用 `return` 提前退出而不是 `elif` 套嵌？

> 两个模式完全隔离——改 RAG 逻辑不会破坏普通聊天，改普通聊天不影响 RAG。`elif` 嵌套会让缩进越来越深，可读性差。

### 1.3 修改 7/7：启动入口

在 `uvicorn.run` 前加了两个 `print`，告诉启动者：
- SQLite 已初始化（messages + documents）
- RAG 懒加载说明（不会阻塞启动）

---

## 二、自提疑问 & 解答

### Q1：删除端点三步的顺序为什么是 ① BM25 → ② Milvus → ③ SQLite？

**背景**：你在看代码时困惑"为什么上传和删除函数有重复内容"，进而追问删除端点的执行顺序逻辑。

**解答**：

这是一个"出错容错"的设计决策。三步的出错影响不同：

| 步骤 | 操作对象 | 失败影响 | 为什么在这个位置 |
|------|---------|---------|----------------|
| ① BM25 | 内存索引 | 无持久数据损失，重启即恢复 | 先清理内存中易丢的数据，失败了不影响持久层 |
| ② Milvus+MinIO | 持久数据 | 如果这步失败，文档还在存储里，重新上传即可 | BM25 已清理干净，不会残留脏索引 |
| ③ SQLite | 元数据 | 最不重要，只是给前端文档列表看的 | 最后删，前面的关键操作都做完再清账本 |

**反例**：如果把 ② 放前面——Milvus 删成功了但 BM25 remove 崩溃了——内存里永远残留已不存在文档的索引，直到下次重启才能清掉。

### Q2：`add_chunks` 和 `remove_by_doc_ids` 都是 O(n) 全量重建，有什么区别？

**背景**：上传用 `add_chunks`，删除用 `remove_by_doc_ids`，看起来都是全量重建 IDF，那为什么不用同一个方法？

**解答**——两种场景的数据来源不同：

```
add_chunks(new_chunks, new_ids):
  已有: self.corpus (旧全量，在内存里)
  新增: new_chunks + new_ids (在参数里)
  操作: corpus + new_chunks → 拼起来 → build_index (重建)
  额外开销: 无（不需要查 Milvus）

remove_by_doc_ids(ids_to_remove):
  已有: self.corpus (旧全量，在内存里)
  要删: ids_to_remove (在参数里，是从 Milvus 查出来的 child_id)
  操作: corpus 过滤掉 ids_to_remove → build_index (重建)
  额外开销: 无（不需要查 Milvus）
```

**都避开了"查 Milvus → 全量 build_index"这种重复网络请求的模式**。

### Q3：为什么上传和删除端点不能用 `MilvusVectorStore()`？

**背景**：我在初版参考代码里写了三次 `vs = MilvusVectorStore()`，你看代码时发现重复创建的问题。

**解答**——因为 `_rag_service` 内部已经有了：

```python
_rag_service                          # RagService 实例
  └── .vector_retriever               # VectorRetriever 实例
        └── .vector_store             # MilvusVectorStore 实例 ← 这个就够了！
```

`new MilvusVectorStore()` 会创建新的 TCP 连接到 Milvus（不管旧连接）。虽然连接成本不高，但"明明钱包里有一张卡还去办新卡"的设计是错误的——多占用一个连接、语义不清晰、让读代码的人困惑"为什么这里要独立连接"。

正确做法：`_rag_service.vector_retriever.vector_store`，一行属性链拿到已有的连接。

### Q4：`:path` 转换器是什么？为什么删除端点要用它？

**背景**：`DELETE /api/documents/{object_key:path}` 这个写法在之前的代码里没见过。

**解答**：

FastAPI 默认的 `str` 类型路径参数会把 URL 里的 `/` 当成路由分隔符：

```
DELETE /api/documents/documents/2026/06/笔记.md
                        └─────str─────┘ ← 只拿到 "documents"
                        剩余 /2026/06/笔记.md → 404 Not Found
```

加了 `:path` 后：

```
DELETE /api/documents/documents/2026/06/笔记.md
                        └──────────path──────────┘ → "documents/2026/06/笔记.md" ✅
```

`object_key` 格式是 `documents/YYYY/MM/filename`，含两个 `/`，所以必须用 `:path`。

---

## 三、踩坑记录

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | 上传/删除端点重复 `new MilvusVectorStore()` | 我没有利用 `_rag_service` 内部已有的实例 | 改为 `_rag_service.vector_retriever.vector_store` |
| 2 | 删除端点用 `build_index(all_chunks)` 从 Milvus 全量重建 | 思路错——不需要再查一次 Milvus，BM25 内部已经有数据 | 改用 `remove_by_doc_ids()` 在内存中过滤+重建 |
| 3 | 删除端点调用 `_rag_service.bm25_index.clear()` | `BM25Index` 没有 `clear()` 方法 | `remove_by_doc_ids()` 内部处理了全删的情况（corpus/doc_ids 置空） |
| 4 | 删除端点的 try 块缺了 `except` | 初版代码漏写了 | 补上 `except Exception as e: print(...)` |
| 5 | `generate_rag_stream_response` 中 `event["type"]` 可能不存在 | `query_stream` 返回的 dict 用 `.get("type")` 更安全 | 已使用 `.get()` |
| 6 | RAG 流式完成时需要保存回复到 DB | 初版漏掉了 `save_message` | 在 `complete` 事件处理中加入 `save_message(session_id, "assistant", accumulated)` |

---

## 四、协作模式评估

**修改 5/7**（C 模式）：我写参考代码，用户看了提出"为什么重复创建"和"删除为什么是重建"两个深入问题。这两个问题暴露了我初版代码的设计缺陷——**C 模式在这里发挥了很好的作用**，用户的困惑变成了改进代码的动力。

**修改 6/7**（C→A 切换）：用户阅读了参考代码后表示"好累"，直接让我写。这个切换是合理的——C 模式 4 个修改下来用户已经理解核心设计模式（属性链复用、BM25 IDF 重建原理、SSE 事件类型），继续 C 模式只会增加疲劳。

**经验**：C 模式适合 3-4 个修改，超过后用户可能疲劳，应及时判断是否切换到 A 模式。

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `step23_rag.py` | ✅ | 全部 7/7 修改完成：14 个 API 端点，RAG 4 模式分流 |
| `rag/indexer.py` | ✅ | 参考读取，确认 DocumentIndexer 接口 |
| `rag/rag_service.py` | ✅ | 参考读取，确认 query() / query_stream() 返回格式 |
| `rag/retrieval/bm25.py` | ✅ | 参考读取，确认 add_chunks() / remove_by_doc_ids() 签名 |
| `rag/retrieval/vector_retriever.py` | ✅ | 参考读取，确认 .vector_store 属性可访问 |

---

## 六、下一步

**Step 12：`static/index.html` 前端集成**

需要在聊天页面增加：
1. 文件上传区域（拖拽/选择文件 → POST /api/documents/upload）
2. 知识库开关（use_rag 切换按钮）
3. 来源文档卡片（收到 type=sources SSE 事件时渲染）
4. 文档列表管理（GET /api/documents + 删除按钮）

**协作模式建议**：A 模式（用户不熟 JavaScript，C 模式不合适）

**前置条件**：Docker 里的 Milvus + MinIO 需要启动才能测试上传和 RAG
