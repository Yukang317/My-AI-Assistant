# indexer.py 骨架说明（临时参考文件，项目完成后删除）

---

## 文件结构

```
rag/indexer.py
│
├── 模块级常量
│   └── CONTENT_TYPE_MAP  ← 文件扩展名 → MIME 类型（MinIO 上传用）
│
├── 模块级工具函数（2 个）
│   ├── _compute_md5(data: bytes) → str         ← 函数1
│   └── _build_object_key(filename: str) → str   ← 函数2
│
└── class DocumentIndexer（4 个方法）
    ├── __init__(self)                            ← 函数3
    ├── index_document(file_data, filename)       ← 函数4 ★核心
    ├── _prepare_chunks_for_milvus(...)           ← 函数5
    └── delete_document(object_key)               ← 函数6
```

---

## 数据流（index_document 内部 8 步）

```
file_data (bytes)
  → _compute_md5(file_data)                          # 计算 MD5
  → _build_object_key(filename)                      # 构建 MinIO 路径
  → self.vector_store.check_document_exists()        # MD5 去重检查
  → self.parser.parse(file_data, file_type)          # Parser: bytes → 纯文本
  → split_by_type(text, doc_type)                    # Chunker: 文本 → 语义块
  → build_parent_child(chunks, filename)             # Chunker: 语义块 → 父子块
  → self.embedder.embed(parent_texts)                # Embed: 父块文本 → 向量
  → self.embedder.embed(child_texts)                 # Embed: 子块文本 → 向量
  → _prepare_chunks_for_milvus(...)                  # 适配: dataclass → dict
  → self.minio.upload(file_data, object_key, type)   # MinIO: 存原始文件
  → self.vector_store.insert_parent_child(...)       # Milvus: 写向量
  → self.vector_store.flush()                        # 强制刷盘
```

---

## Insight 要点

1. **编排者的核心价值在"顺序契约"**：`index_document` 自己一行解析逻辑都不写，但它规定了每一步的输入输出类型——调用方不需要知道 parser/chunker/embedder 的内部细节，只需知道"把 bytes 给我，我把索引好的结果还给你"。

2. **`_prepare_chunks_for_milvus` 是一个"数据适配层"**：上游产出的是 `ParentChunk`/`ChildChunk` dataclass（语义清晰），下游期望的是 `Dict[str, Any]`（Milvus 只认识这个）。中间夹一层转换，两边都不需要为对方妥协自己的数据结构。

3. **MD5 去重不在 parser 也不在 vector_store**——它在 `index_document` 里。因为"判断一个文件是否已索引"需要同时看 MinIO（有没有存过）和 Milvus（有没有向量），这是编排者独有的全局视角，任何一个子模块都看不到全貌。

---

## 框架引入时机（"疼了才引入"）

| Step | 模块 | 框架 |
|------|------|------|
| 1-5 | config / embedding / minio / vector_store / parser | ❌ 纯原生 |
| 6 | chunker | ✅ LangChain 分块器 |
| **7** | **indexer（编排者）** | **❌ 纯原生** |
| 8 | retrieval | ⚠️ 先手写 RRF，后升级 LlamaIndex |
| 9 | rag_service | ✅ LangChain |
