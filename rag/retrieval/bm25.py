"""
BM25 关键词检索器

基于 rank_bm25.BM25Okapi 算法，使用 jieba 分词支持中文。
BM25 与向量检索互补：
  - 向量检索：语义相似，能理解"电脑"≈"计算机"
  - BM25 检索：精确匹配，能命中 API 名、术语、代码片段等

使用方式：
  bm25 = BM25Index()
  bm25.build_index(chunks=["文档1内容...", "文档2内容..."], doc_ids=["doc1", "doc2"])
  results = bm25.search("DeepSeek API", top_k=5)
  # → [{"chunk_index": 0, "doc_id": "doc1", "score": 1.52, "text": "文档1内容..."}, ...]
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BM25Index:
    """文件级 BM25 关键词索引

    内部使用 Okapi BM25 算法（k1=1.5, b=0.75），jieba 分词支持中文。
    注意：BM25 的 IDF 依赖全量语料库，增删操作会触发全量重建。

    Attributes:
        language: 分词语言，"zh" 使用 jieba，其他使用空格分词
        corpus: 所有 chunk 的原始文本列表
        doc_ids: 与 corpus 一一对应的文档标识列表
        _bm25_model: BM25Okapi 实例（build_index 后可用）
    """

    def __init__(self, language: str = "zh"):
        """初始化 BM25 索引

        Args:
            language: 分词语言，"zh" 启用 jieba 中文分词，
                      其他值使用空格分词（适用于英文）。
        """
        # TODO(human): 初始化 self.language, self.corpus, self.doc_ids, self._bm25_model
        self.language = language
        self.corpus = []
        self.doc_ids = []
        self._bm25_model = None

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词

        中文模式使用 jieba.cut（精确模式），英文模式使用 str.split()。
        分词结果会过滤掉空字符串。

        Args:
            text: 待分词的原始文本

        Returns:
            分词后的 token 列表，例如 ["DeepSeek", "API", "怎么", "调用"]
        """
        # TODO(human): 根据 self.language 选择分词方式，过滤空 token

    # ── 核心操作 ──────────────────────────────────────────────────

    def build_index(self, chunks: List[str], doc_ids: List[str]) -> None:
        """全量构建 BM25 索引

        对每个 chunk 进行分词，然后用全部 tokenized corpus 构建 BM25Okapi 模型。
        构建后可通过 search() 检索。

        Args:
            chunks: 所有 chunk 的文本列表，长度 = N
            doc_ids: 每个 chunk 对应的文档标识（如 MinIO object_key），长度必须 = N

        Raises:
            ValueError: 如果 len(chunks) != len(doc_ids) 或 chunks 为空
        """
        # TODO(human): 验证参数 → 对每个 chunk 分词 → 构建 BM25Okapi → 保存 corpus/doc_ids

    def search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """BM25 关键词检索

        对查询分词后，用 BM25 算法计算与每个 chunk 的相关性分数，
        返回 top_k 个最高分结果。

        Args:
            query: 用户查询文本
            top_k: 返回结果数量，默认 5

        Returns:
            按分数降序排列的结果列表，每项包含:
            - chunk_index: int, 在 self.corpus 中的索引
            - doc_id: str, 文档标识
            - score: float, BM25 相关性分数（越高越相关）
            - text: str, chunk 原始文本

        Raises:
            RuntimeError: 如果索引尚未构建（需先调用 build_index）
        """
        # TODO(human): 检查索引状态 → 对 query 分词 → get_scores → 取 top_k → 组装返回

    # ── 增量维护 ──────────────────────────────────────────────────

    def add_chunks(self, chunks: List[str], doc_ids: List[str]) -> None:
        """增量添加 chunk（追加到现有 corpus 后重建索引）

        由于 BM25 的 IDF 依赖全量语料库，添加后必须重建模型。

        Args:
            chunks: 新增的 chunk 文本列表
            doc_ids: 对应的文档标识列表，长度必须与 chunks 一致

        Raises:
            ValueError: 参数长度不匹配
            RuntimeError: 如果索引尚未初始化（需先调用 build_index）
        """
        # TODO(human): 检查索引状态 → 验证参数 → 追加到 corpus/doc_ids → 重建模型

    def remove_by_doc_ids(self, doc_ids_to_remove: List[str]) -> int:
        """按文档 ID 批量删除 chunk（过滤后重建索引）

        Args:
            doc_ids_to_remove: 要删除的文档标识列表

        Returns:
            实际删除的 chunk 数量

        Raises:
            RuntimeError: 如果索引尚未构建
        """
        # TODO(human): 检查索引状态 → 过滤 corpus/doc_ids → 重建模型 → 返回删除数量

    # ── 状态查询 ──────────────────────────────────────────────────

    def is_built(self) -> bool:
        """检查索引是否已构建

        Returns:
            True 如果已调用 build_index 并成功构建
        """
        # TODO(human): 返回 self._bm25_model 是否为 None

    def get_stats(self) -> Dict[str, int]:
        """获取索引统计信息

        Returns:
            {"chunk_count": N, "unique_docs": M}
        """
        # TODO(human): 返回 chunk 总数和唯一文档数
