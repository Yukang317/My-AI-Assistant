# 检索管线：BM25 关键词 + 向量语义 → RRF 融合 → 重排序 → 父块回溯
# 借鉴 LlamaIndex QueryFusionRetriever + SentenceTransformerRerank
#
# 检索流程：
#   bm25_index.search() + vector_store.search() → hybrid_searcher.rrf_fusion()
#   → reranker.rerank() → 父块回溯 → 返回最终上下文
