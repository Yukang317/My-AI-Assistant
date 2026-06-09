"""
向量语义检索器

封装"查询 → Embedding → Milvus 搜索"的完整链路。
与 bm25.py 是平行的检索通道：
  - BM25：关键词精确匹配（适合 API 名、术语、代码片段）
  - Vector：语义相似匹配（适合"电脑"≈"计算机"这种同义表达）

使用方式：
  from rag.embedding import EmbeddingService
  from rag.storage.vector_store import MilvusVectorStore
  from rag.retrieval.vector_retriever import VectorRetriever

  vs = MilvusVectorStore()
  emb = EmbeddingService()
  retriever = VectorRetriever(emb, vs)
  results = retriever.search("DeepSeek 怎么调？", top_k=10)
  # → [{"chunk_id": "报告_A_p0_c3", "doc_id": "documents/deepseek.md",
  #     "score": 0.92, "text": "DeepSeek API 调用方式...",
  #     "parent_text": "完整父块上下文（2048 tokens）...", ...}, ...]
"""

import logging
from typing import Any, Dict, List, Optional

from rag.embedding import EmbeddingService
from rag.storage.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)


class VectorRetriever:
    """向量语义检索器 — 对查询文本做 embedding，到 Milvus 做 ANN 搜索。

    依赖注入 EmbeddingService 和 MilvusVectorStore，
    不自己管理连接，只做协调编排。

    Attributes:
        embedding_service: EmbeddingService 实例（用于将查询文本转为向量）
        vector_store: MilvusVectorStore 实例（用于 ANN 搜索和父块回溯）
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: MilvusVectorStore,
    ) -> None:
        """初始化向量检索器。

        两个依赖从外部注入，这样：
          - 多个检索器可以共享同一个 embedding 模型（省内存）
          - 测试时可以 mock 掉 Milvus

        Args:
            embedding_service: 已初始化的 EmbeddingService 实例
            vector_store: 已连接 Milvus 的 MilvusVectorStore 实例
        """
        # TODO(human): 把两个依赖存到 self 上
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    # ── 私有方法 ──────────────────────────────────────────────────

    def _embed_query(self, query: str) -> List[float]:
        """将单条查询文本转为向量。

        封装 embedding_service.embed() 调用，处理：
          - embed() 接收列表返回列表，query 是单字符串需要包装
          - 取返回列表的第一个元素（因为只传了一个查询）
          - 空查询的防御

        Args:
            query: 用户查询文本

        Returns:
            1024 维浮点数向量（已 L2 归一化）

        Raises:
            ValueError: query 为空字符串时抛出
        """
        # TODO(human): 1. 检查 query 非空
        # 2. 调用 self.embedding_service.embed([query])（注意包装成列表）
        # 3. 返回第一个（也是唯一一个）向量
        if not query.strip():
            raise ValueError("query 不能为空")
        query_vectors = self.embedding_service.embed([query])  # 得到 [[0.1, 0.2, ...]]

        return query_vectors[0] # 拆出内层

    # ── 核心检索 ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """语义检索主入口：查询 → 向量化 → Milvus ANN 搜索。

        完整链路：
          1. _embed_query(query)                  # 文本 → 1024 维向量
          2. vector_store.search(vector, top_k)    # ANN 搜索 + 父块回溯
          3. 格式化为统一输出                        # 对齐 BM25 search() 返回格式

        Args:
            query: 用户查询文本
            top_k: 返回的最相似结果数，默认 10
            filter_expr: Milvus 标量过滤表达式，如 'doc_type == "md"'
                         传 None 则不过滤（搜索全部文件类型）

        Returns:
            按 score 降序排列的结果列表，每项包含:
            - chunk_id: str, 子块业务 ID（如 "报告_A_p0_c3"）
            - doc_id: str, 文档路径（MinIO object_key）
            - doc_name: str, 文档名
            - doc_type: str, 文件类型
            - score: float, 相似度分数（IP 内积，越高越相似）
            - text: str, 子块文本（512 tokens，精准检索用）
            - parent_text: str, 父块文本（2048 tokens，提供完整上下文）

        Raises:
            ValueError: query 为空时抛出
            RuntimeError: Milvus 搜索失败时抛出
        """
        # TODO(human): 1. _embed_query 向量化
        # 2. 调 vector_store.search(query_vector, top_k, filter_expr)
        # 3. 把 vector_store 返回的格式转成统一输出格式
        #    vector_store 返回格式见 vector_store.py 第 330-342 行的注释
        #    需要做字段映射：
        #      child_id      → chunk_id
        #      doc_path_name → doc_id
        #      child_content → text
        #      parent_content → parent_text
        #      score/doc_name/doc_type → 原样保留
        # 4. 返回结果列表
        # 拿到查询向量
        query_vector = self._embed_query(query)

        # 拿到Milvus的结果，格式是：[{child_id, parent_id, child_content, parent_content, doc_name, doc_path_name, doc_type, score}, ...]
        raw_results = self.vector_store.search(query_vector, top_k, filter_expr)
        results = []

        for item in raw_results:
            result = {
                # 映射转换的部分
                "chunk_id":item["child_id"],
                "doc_id": item["doc_path_name"], 
                "text": item["child_content"],
                "parent_text": item["parent_content"],
                # 保留不变的部分
                "doc_name": item["doc_name"],
                "doc_type": item["doc_type"],
                "score": item["score"],

            }
            results.append(result)
        
        return results
        






    # ── 便利方法 ──────────────────────────────────────────────────

    def search_by_type(
        self,
        query: str,
        doc_type: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """按文件类型过滤的语义检索（便利方法）。

        等价于 search(query, top_k, filter_expr='doc_type == "{doc_type}"')

        Args:
            query: 用户查询文本
            doc_type: 文件类型（"md", "pdf", "docx", "txt"）
            top_k: 返回结果数

        Returns:
            与 search() 相同的格式
        """
        # TODO(human): 构建 filter_expr，然后委托给 self.search()
        filter_expr = f'doc_type == "{doc_type}"'
        return self.search(query, top_k, filter_expr)

    # ── 状态查询 ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取检索器统计信息。

        从 vector_store 拉取 Milvus Collection 信息。

        Returns:
            {
                "embedding_mode": "local",
                "embedding_model": "BAAI/bge-large-zh-v1.5",
                "parent_collection": "parent_chunks",
                "child_collection": "child_chunks",
            }
        """
        # TODO(human): 返回 embedding_service 和 vector_store 的关键配置
        embedding_mode = self.embedding_service.mode    # mode是属性不是方法
        embedding_model = Config.get_embedding_model_name()
        parent_collection = Config.MILVUS_PARENT_COLLECTION
        child_collection = Config.MILVUS_CHILD_COLLECTION
        return {
            "embedding_mode": embedding_mode,  # local 或 api
            "embedding_model": embedding_model,
            "parent_collection": parent_collection,
            "child_collection": child_collection,
        }
