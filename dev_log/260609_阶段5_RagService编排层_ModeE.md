# 阶段 5 — RagService 编排层（rag/rag_service.py）

**日期**：2026-06-09
**模式**：E（骨架式教学开发，修订版）
**内容**：完成 RAG 服务编排层 8 个函数，检索管线全链路闭环

---

## 〇、做了什么 & 为什么

**当前在项目中的位置**：路线2（知识检索）进入最后一环。BM25、VectorRetriever、RRFFusion 三个检索组件就绪后，需要一个顶层编排者把它们串联成"查询 → 检索 → LLM 生成"的完整管线。

**本次完成**：`rag/rag_service.py` — RagService 类，8 个函数。

**在整体架构中的作用**：RagService 是路线2的"大脑"，对上暴露 `query()` 和 `query_stream()` 两个公开方法给 FastAPI 端点调用，对内调度 BM25 + Vector → RRF → 重排 → 上下文拼接 → LLM 生成。

---

## 一、代码全貌

### 类结构

```
RagService
├── __init__()           # 依赖注入 + LLM 客户端
├── _retrieve()          # BM25 + Vector → RRF 融合（私有）
├── _rerank()            # 重排序（USE_RERANK=False 时透传）
├── _build_context()     # 文档列表 → 格式化文本
├── _build_prompt()      # 填充 Config 模板
├── query()              # 同步管线入口
├── query_stream()       # 异步流式入口（SSE）
└── get_stats()          # 健康检查汇总
```

### __init__ — 依赖注入 + 双 LLM 客户端

```python
def __init__(self, bm25_index, vector_retriever, rrf_fusion, llm_client=None):
    self.bm25_index = bm25_index          # 外部注入，不自己创建
    self.vector_retriever = vector_retriever
    self.rrf_fusion = rrf_fusion
    self.llm_model = Config.LLM_MODEL     # ← 容易漏！
    self.llm_client = llm_client or OpenAI(...)             # 同步客户端
    self.llm_async_client = AsyncOpenAI(...)                 # 流式客户端
    self.use_rerank = Config.USE_RERANK   # 默认 False
```

关键点：`llm_async_client` 必须是 `AsyncOpenAI`（不是 `OpenAI`），因为 `query_stream()` 里用 `await` + `stream=True`。

### _retrieve — 双通道检索 + RRF 融合

```python
def _retrieve(self, query, top_k=10, filter_expr=None):
    bm25_results = self.bm25_index.search(query, top_k=top_k)
    vector_results = self.vector_retriever.search(query, top_k=top_k, filter_expr=filter_expr)
    return self.rrf_fusion.fuse(bm25_results, vector_results, top_k=top_k)
```

三个检索组件各司其职：BM25 做关键词精确匹配，Vector 做语义相似搜索，RRF 用排名融合（不关心原始分数尺度差异）。

### _rerank — 重排序（当前透传）

```python
def _rerank(self, query, documents, top_n=5):
    if not self.use_rerank:
        return documents[:top_n]  # 当前走这个分支
    # --- 以下是重排开启后的逻辑（USE_RERANK=True 时执行）---
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("BAAI/bge-reranker-large")
    pairs = [[query, doc["text"]] for doc in documents]
    scores = model.predict(pairs)
    # ... 排序 + 安全兜底（向量第1名强制保留）...
```

`USE_RERANK=False` 的原因：bge-reranker-large 约 1.3GB，ECS 3.5GB 内存上不能和 BGE embedding 模型同时驻留。基础管线跑通后再开。

### _build_context — 文档列表 → LLM 可读文本

```python
def _build_context(self, documents):
    if not documents:
        return "知识库中暂无相关内容。"
    parts = []
    for doc in documents:
        content = doc.get("parent_text") or doc.get("text", "")  # 父块优先
        doc_name = doc.get("doc_name", "未知文档")
        doc_type = doc.get("doc_type", "")
        part = f"[来源：{doc_name} ({doc_type})]\n{content}\n---"
        parts.append(part)
    return "\n\n".join(parts)
```

用 `.get()` 不用中括号——纯 BM25 命中的文档没有 `parent_text` 字段，中括号会 KeyError。

### _build_prompt — 模板填充

```python
def _build_prompt(self, question, context):
    return (
        Config.RAG_PROMPT_TEMPLATE
        .replace("{context}", context)
        .replace("{question}", question)
    )
```

用 `.replace()` 不用 `.format()`——模板中可能含 Markdown 代码块和 JSON 的花括号。

### query — 同步管线（9 步）

```python
def query(self, question, top_k=10, filter_expr=None, conversation_history=None):
    # 1. 验证 → 2. 计时开始 → 3. _retrieve + _rerank
    # → 4. _build_context + _build_prompt → 5. LLM 调用
    # → 6-7. 提取 answer + token_usage → 8. 延迟计算 → 9. 返回
    return {"answer": ..., "sources": ..., "token_usage": ..., "latency_ms": ..., "question": ...}
```

### query_stream — 异步流式（SSE 四种事件）

```python
async def query_stream(self, question, ...):
    try:
        # 检索 + 重排 + Prompt 构建（和 query 一样）
        yield {"type": "sources", "documents": final_docs}      # ① 先推来源
        stream = await self.llm_async_client.chat.completions.create(stream=True)
        async for chunk in stream:
            yield {"type": "delta", "content": delta.content}   # ② 逐字推送
        yield {"type": "complete", "token_usage": {}}           # ③ 结束信号
    except Exception as e:
        yield {"type": "error", "content": str(e)}              # ④ 错误保护
```

`try` 包裹全部逻辑——检索阶段出错也不会让前端收到 HTTP 500，而是优雅的 SSE error 事件。

### get_stats — 健康检查汇总

```python
def get_stats(self):
    return {
        "bm25": self.bm25_index.get_stats(),
        "vector": self.vector_retriever.get_stats(),
        "rrf": self.rrf_fusion.get_stats(),
        "use_rerank": self.use_rerank,
        "llm_model": self.llm_model,
    }
```

各子组件自带 `get_stats()`，RagService 只做汇总不解析——不改动时不用改顶层代码。

---

## 二、自提疑问 & 解答

### Q1：为什么 BM25 和 Vector 的 top_k 用一样的值，不是说 BM25 精确度高应该取少一点吗？

**解答**：`_retrieve` 是私有方法，职责是简单透传，不做精细化控制。如果后续需要 BM25 取 5、Vector 取 10，可以在公开的 `query()` 里直接调 `self.bm25_index.search()` 和 `self.vector_retriever.search()` 传不同的 top_k，绕过 `_retrieve`。当前阶段用一个 top_k 够用。

### Q2：`[来源：{doc_name} ({doc_type})]` 这个格式在实际使用中是怎么体现在 Prompt 里的？

**解答**：`_build_context` 输出 → `_build_prompt` 填入 `{context}` 占位符 → 发给 LLM 的最终 Prompt 长这样：

```
你是一个个人 AI 助理。请根据用户的知识库内容回答问题。

知识库内容:
[来源：DeepSeek笔记 (md)]
DeepSeek API 的调用方式和 OpenAI 完全兼容。base_url 设为 https://api.deepseek.com/v1...
---

[来源：Python笔记 (md)]
Python 中使用 openai 包调用 LLM...
---

用户问题: DeepSeek API 怎么调？
```

LLM 看到 `[来源：...]` 标签就知道每段来自哪个文件，回答末尾才能"标注来源文档"。

### Q3：`query()` 里 `model=self.llm_model` 是从哪来的？

**解答**：`self.llm_model = Config.LLM_MODEL` 在 `__init__` 里设置，值是 `"deepseek-chat"`。如果不写这行，运行时直接 `AttributeError: 'RagService' object has no attribute 'llm_model'`——这是本次踩到的一个 bug。

---

## 三、踩坑记录

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `Config.DEEPSEEK_API_KEY` 报 AttributeError | Config 类属性是 `LLM_API_KEY`，不是 `DEEPSEEK_API_KEY` | 改为 `Config.LLM_API_KEY` |
| 2 | `Config.DEEPSEEK_API_URL` 不存在 | Config 类属性是 `LLM_BASE_URL` | 改为 `Config.LLM_BASE_URL` |
| 3 | `_rerank` 中 `CrossEncoders` 拼写错误 | 多了一个 s，应为 `CrossEncoder` | 改为 `CrossEncoder` |
| 4 | `query_stream` 中 `await self.llm_client.chat...` 报错 | `OpenAI` 是同步客户端，`await` + `stream=True` 需要 `AsyncOpenAI` | `__init__` 里创建 `self.llm_async_client = AsyncOpenAI(...)` |
| 5 | `query()` 报 `self.llm_model` 不存在 | `__init__` 漏了 `self.llm_model = Config.LLM_MODEL` | 在 `__init__` 里加上这一行 |
| 6 | `query_stream` 中 `_build_prompt` 行语法错误 | `_build_prompt(question=question, context=context` 少了右括号 | 补上 `)` |
| 7 | `VectorRetriever()` 无参兜底会 TypeError | 它需要 `embedding_service` 和 `vector_store` 两个必传参数 | 去掉多余的 `or VectorRetriever()` 兜底，直接赋值 |

---

## 四、协作模式评估

### Mode E 修订（本次会话重要变更）

本次会话中升级了 Mode E 的 Step 2 逐函数教学流程：

**改动前**：知识点讲解可以堆砌专业名词，代码提示仅给出思路，用户自己摸索实现。

**改动后**：
1. 知识点讲解**必须结合项目具体流程**解释每个概念，说明"在这个项目里数据怎么流"，禁止用抽象专业名词定义（如"依赖注入是一种实现了控制反转的设计模式"）
2. **在回复中直接给出参考实现**（完整示例代码 + 逐行注释），用户看过后自己动手写入文件
3. 禁止直接编辑代码文件

**效果**：用户实现速度明显加快，bug 数量和之前模块相比大幅减少（7 个，主要是 Config 属性名和类型混用，没有逻辑错误）。

**相关文件更新**：`CLAUDE.md` 和 `阶段5技术方案.md` 已同步更新。

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/rag_service.py` | ✅ | 8 函数全部完成，检索管线闭环 |
| `CLAUDE.md` | ✅ | Mode E 流程已升级 |
| `阶段5技术方案.md` | ✅ | 协作模式表格已同步 |

---

## 六、下一步

**路线2（知识检索）还剩最后一个文件：`rag/evaluate.py`**

```
路线1（文档摄入）✅ 全部完成！
路线2（知识检索）🔄 
✅ Step 8a: rag/retrieval/bm25.py
✅ Step 8b: rag/retrieval/vector_retriever.py
✅ Step 8c: rag/retrieval/rrf_fusion.py
✅ Step 9:  rag/rag_service.py              ← 刚完成！
⬜ Step 10: rag/evaluate.py                  ← 下一个
```

**下一文件**：`rag/evaluate.py` — RAG 评估（忠实度/召回率/精确率）
**协作模式**：Mode E（骨架式教学开发）
