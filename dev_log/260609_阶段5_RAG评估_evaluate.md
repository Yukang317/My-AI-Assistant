# 阶段 5 — RAG 评估模块（rag/evaluate.py）

**日期**：2026-06-09
**模式**：E（骨架式教学开发）
**内容**：基于 RAGAS 框架的 RAG 评估模块，自动出题 + 自动打分，完成路线2（知识检索）最后一块拼图

---

## 〇、做了什么 & 为什么

**当前在项目中的位置**：路线2（知识检索）的最后一个文件。BM25、VectorRetriever、RRFFusion、RagService 全部就绪后，需要一个评估模块来定量衡量检索链路的性能——"查得准不准、答案有没有编造"。

**本次完成**：`rag/evaluate.py` — 9 个函数，5 步评估管线。

**在整体架构中的作用**：评估模块是独立于业务管线的"质检层"。它不参与用户查询，而是通过 RAGAS 框架自动从知识库出题、调 RagService 获取回答、用 LLM-as-judge 打分，输出 JSON 报告。是后续调优 Prompt / chunk_size / top_k 等参数的量化依据。

**核心决策**：**用 RAGAS 框架而非手写评估算法**。用户明确指出"没有精力写测试数据、也不知道正确答案"——这正是 RAGAS TestsetGenerator 解决的问题（从文档自动生成问题和标准答案）。`evaluate.py` 只做编排层，核心逻辑全部委托给 RAGAS。

---

## 一、代码全貌

### 整体架构

```
evaluate.py（薄编排层，~490 行）
├── _build_ragas_llm()              # RAGAS LLM 客户端（DeepSeek）
├── _build_ragas_embeddings()       # RAGAS Embedding 客户端（BGE 本地）
├── _load_parent_docs_from_milvus() # Milvus 随机采样 → 父块回溯
├── generate_testset()              # RAGAS 自动出题
├── _init_rag_service()             # RagService 完整初始化
├── evaluate_rag()                  # 调 RagService + 自动打分
├── save_report()                   # 保存 JSON 报告
├── print_summary()                 # 控制台摘要
└── main()                          # CLI 入口（5 步串联）
```

### 评估管线数据流

```
① Milvus child_chunks 随机向量搜索 → parent_id 回溯 parent_chunks
    → 去重父块 → 包成 LangChain Document（page_content=父块全文）

② RAGAS TestsetGenerator.generate_with_langchain_docs(docs, testset_size=5)
    → LLM 分析文档 → 生成问题 → 生成标准答案 → 自洽性检查
    → Dataset {question, ground_truth, contexts}

③ 遍历 testset → rag_service.query(question)
    → 收集 answers（回答文本）+ contexts（检索到的父块文本）

④ ragas.evaluate(dataset, metrics=[4项], llm=, embeddings=)
    → LLM-as-judge 逐项打分
    → 返回逐样本分数 → 计算均值

⑤ 保存 eval_results/eval_report_{timestamp}.json + 打印摘要
```

### 关键函数要点

**`_load_parent_docs_from_milvus`** — 随机采样核心：
- 用 `np.random.randn(1024).tolist()` 生成随机向量，在 child_chunks 中搜索
- `vs.search()` 返回结果已含 `parent_content`（search 内部自动做了父块回溯），无需手动查
- 按 `parent_id` 去重（多个子块可能属于同一父块）

**`generate_testset`** — RAGAS 0.4.x API 集成：
- `TestsetGenerator.from_langchain(llm, embedding_model=embeddings)` — 注意参数名
- `generate_with_langchain_docs(docs, testset_size=N, query_distribution={...})` — 注意参数名
- 三类问题：simple 0.4, reasoning 0.3, multi_context 0.3

**`_init_rag_service`** — 全量 BM25 构建：
- 通过 `vs.client.query()` 查询所有 child_chunks（limit=2000），构建 BM25 全量索引
- EmbeddingService 最后创建（加载 BGE ~1.3GB 内存）

**`evaluate_rag`** — 评估主循环：
- `ctx = [s.get("parent_text") or s.get("text", "") for s in sources]` — 优先父块，用 `or` 兜底
- NaN/Inf → None 转换（`math.isnan`/`math.isinf`），保证 JSON 序列化
- 计算均值时过滤 None 值

---

## 二、自提疑问 & 解答

### Q1：为什么 RAGAS 出题要 5 条而不是更多？

**解答**：每条测试样本需要 2-3 次 LLM 调用（出题→出答案→验证自洽），5 条约 10-15 次调用，耗时约 1-2 分钟。考虑到：① 这是首次搭建评估管线，先跑通再增量；② DeepSeek API 按 token 计费；③ RAGAS 的 TestsetGenerator 每次生成结果略有差异（即使 temperature=0），少一点反而便于快速迭代对比。后续稳定后可提升到 10-20 条。

### Q2：为什么不直接从 MinIO 读原始文件给 RAGAS？

**解答**：那个路线（MinIO → Parser → LangChain Document → RAGAS）多了一步"全文解析"，而 RAGAS 出题其实只需要文档片段有语义完整性。我们的父子分块结构里，父块是 2048 字符的完整语义片段，完全满足 RAGAS 出题需求。从 Milvus 采样比走 MinIO→Parser 快得多，且不需要担心 PDF/Word 等格式的解析开销。

### Q3：BM25 为什么要全量构建，而不能只从采样的 5 条文档构建？

**解答**：BM25 的 IDF（逆文档频率）依赖全局统计——它需要知道每个词在多少文档中出现才能算出正确的权重。如果只用 5 条文档构建，RAGAS 生成的测试问题中的关键词可能不在那些文档里，BM25 检索就完全废了。必须把所有 child_chunks 都喂进 BM25，才能保证检索覆盖面和评分公平性。

### Q4：`_build_ragas_embeddings` 和 `EmbeddingService` 都加载 BGE，内存不会爆吗？

**解答**：`sentence-transformers` 对同一个模型路径有内置缓存——`SentenceTransformer("BAAI/bge-large-zh-v1.5")` 第二次调用时不会重复加载权重，只做类型检查。所以 RAGAS 的 `HuggingFaceEmbeddings` 和 `EmbeddingService` 实际上共享同一个底层模型，内存占用 ~1.3GB 不变。前提是两次调用都指定同一个模型名。

---

## 三、踩坑记录

| # | Bug 现象 | 根因 | 修复 |
|---|---------|------|------|
| 1 | `cleaned = {k: sanitize(v) for k, v in result.items()}` 会 `AttributeError` | `result` 是 `list[dict]` 不是 `dict`，对列表调 `.items()` 报错；这是遗留代码，下面的 `cleaned_rows` 循环才是正确实现 | 注释掉该行 |
| 2 | `print(f"采样完成...")` 缩进错误——文档数够时被跳过 | 多缩进了 4 空格，嵌在 `if len(docs) < test_size:` 里 | 将缩进改回与 `if` 同级 |
| 3 | pymilvus 未安装在 `.venv` 里 | `uv sync` 没把 pymilvus 拉进来（milvus 之前可能是在系统或别的 venv 装的）| `uv add pymilvus` |
| 4 | 网络超时——uv sync 下载 numpy/scipy/pandas 等大包反复超时 | PyPI 直连在国内不稳定，好几个包超过 30MB | `UV_HTTP_TIMEOUT=600 uv sync`，或配置清华镜像源 |
| 5 | `generate_with_langchain_docs` 参数名写错 | RAGAS 0.4.x API 是 `testset_size` 不是 `test_size`，`query_distribution` 不是 `distributions` | 对照 RAGAS 文档和 MilDoc 参考代码修正 |

---

## 四、协作模式评估

### Mode E 用得怎么样

本次是 Mode E（骨架式教学开发）的典型应用：
- 骨架文件 9 个函数，全部编译通过后逐函数教学
- 用户实现了 2 个 bug（328 行遗留代码、459 行缩进），恰好都是"遗留清理"类而非逻辑错误——说明参考实现起到了引导作用
- `临时需要的内容.md` 里的笔记质量很好——包含了代码逻辑和架构理解，不只是照搬

### 改进观察
- 用户在笔记中对"数据处理技巧"（NaN 清洗、字典遍历 + labels 映射）的提炼很精准——这些不是业务逻辑，是写代码的技巧，说明 Mode E 的教学效果在从"写什么"过渡到"怎么写得好"

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/evaluate.py` | ✅ | 9 函数全部完成，路线2 检索全线闭环 |
| `临时需要的内容.md` | ✅ | 用户学习笔记（环境配置 + RAGAS 概念 + 代码逻辑） |
| `pyproject.toml` | ✅ | 新增 ragas, datasets, langchain*, pandas 依赖 |

---

## 六、下一步

```
路线2（知识检索）✅ 全部完成！
   ✅ Step 8a: rag/retrieval/bm25.py
   ✅ Step 8b: rag/retrieval/vector_retriever.py
   ✅ Step 8c: rag/retrieval/rrf_fusion.py
   ✅ Step 9:  rag/rag_service.py
   ✅ Step 10: rag/evaluate.py          ← 刚完成！

路线3（整合+前端）⬜ 开始
   ⬜ Step 11: FastAPI 端点（/api/chat/rag, /api/chat/rag/stream）
   ⬜ Step 12: 前端集成 + Docker 启动
```

**下一文件**：`step22_session.py` 改造（新增 RAG 端点）或 `static/index.html` 集成 RAG 对话
**协作模式**：待定（首次写 FastAPI 端点可用 Mode E，前端 JS 可用 Mode A）

**运行评估**：
```bash
cd personal_assistant && .venv/bin/python -m rag.evaluate
```
