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
from rank_bm25 import BM25Okapi
import jieba
import numpy as np


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
        # 翻译： corpus：语料库 —— 即一组文档、文本数据的集合，用于检索或训练模型
        self.corpus = []
        self.doc_ids = []
        self._bm25_model = None

    def _tokenize(self, text: str) -> List[str]:
        """对文本进行分词，处理单个 chunk 字符串

        中文模式使用 jieba.cut（精确模式），英文模式使用 str.split()。
        分词结果会过滤掉空字符串。

        Args:
            text: 待分词的原始文本

        Returns:
            分词后的 token 列表，例如 ["DeepSeek", "API", "怎么", "调用"]
        """
        # eg：["DeepSeek", "API"]
        # TODO(human): 根据 self.language 选择分词方式，过滤空 token
        if self.language == "zh":
            # jieba 精确模式分词。中文
            tokens = [t for t in jieba.cut(text) if t.strip()]  # if t.strip() 等价于 if t.strip() != ""
        else:
            # 空格分词。英文
            tokens = [t for t in text.split() if t.strip()]

        return tokens

    # ── 核心操作 ──────────────────────────────────────────────────

    def build_index(self, chunks: List[str], doc_ids: List[str]) -> None:
        """全量构建 BM25 索引，处理全部 chunk 集合

        对每个 chunk 进行分词，然后用全部 tokenized corpus 构建 BM25Okapi 模型。
        构建后可通过 search() 检索。

        Args:
            chunks: 所有 chunk 的文本列表，长度 = N
            doc_ids: 每个 chunk 对应的文档标识（如 MinIO object_key），长度必须 = N

        Raises:
            ValueError: 如果 len(chunks) != len(doc_ids) 或 chunks 为空
        """
        # TODO(human): 验证参数 → 对每个 chunk 分词 → 构建 BM25Okapi → 保存 corpus/doc_ids
        # 1. 参数验证
        if not chunks:
            raise ValueError("chunks 不能为空")
        
        if len(chunks) != len(doc_ids):
            raise ValueError(f"chunks : {len(chunks)} 和 doc_ids : {len(doc_ids)} 的长度不匹配")
        
        # 2. 分词 -> 构建 BM25模型
        tokenized = [self._tokenize(chunk) for chunk in chunks]
        self._bm25_model = BM25Okapi(tokenized)

        # 3. 保存
        self.corpus = list(chunks)
        self.doc_ids = list(doc_ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
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
        # 1. 检查索引是否构建
        if not self.is_built():
            raise RuntimeError("索引尚未构建，请先调用 build_index")

        # 2. 查询分词
        tokenized_query = self._tokenize(query)

        # 3. 获取所有分数
        score = np.array(self._bm25_model.get_scores(tokenized_query))

        # 4. argpartition 取 top_k 个最高分的索引
        k = min(top_k, len(score))
        # <= 第 k 个元素的，都在左边，左右两边内部的顺序是乱的！
        # top_indices 是找到的前k个最高分文档的索引（但未排序）
        # 比如: [3, 5, 1, 8, 2]
        top_indices = np.argpartition(score, -k)[-k:]  
        # score[top_indices] 就是取出这些索引对应的分数
        # 比如: [0.9, 0.7, 0.5, 0.85, 0.45]    
        top_indices = top_indices[np.argsort(score[top_indices])[::-1]] # 按分数降序排。[::-1]：因为 argsort 默认是升序，加个倒转变成降序

        # 5. 组装返回
        results = []
        for idx in top_indices:
            if score[idx] > 0:
                # int(idx) 和 float(scores[idx]) —— numpy 类型转回 Python 原生类型，避免 JSON序列化时出问题。
                results.append({
                    "chunk_index": int(idx),
                    "doc_id": self.doc_ids[idx],
                    "score": float(score[idx]),
                    "text": self.corpus[idx],
                })

        # 6. 返回结果
        return results


    # ── 增量维护 ──────────────────────────────────────────────────

    def add_chunks(self, chunks: List[str], doc_ids: List[str]) -> None:
        """== 用于每次用户新添加文件的场景 ==
        增量添加 chunk（追加到现有 corpus 后重建索引）。

        由于 BM25 的 IDF 依赖全量语料库，添加后必须重建模型。
        已有的数据在 self.corpus 和 self.doc_ids 里，新来的数据在参数 chunks 和 doc_ids
        里。把它们拼起来，然后做和 build_index 一模一样的事。

        Args:
            chunks: 新增的 chunk 文本列表
            doc_ids: 新增对应的文档标识列表，长度必须与 chunks 一致

        Raises:
            ValueError: 参数长度不匹配
            RuntimeError: 如果索引尚未初始化（需先调用 build_index）
        """
        # TODO(human): 检查索引状态 → 验证参数 → 追加到 corpus/doc_ids → 重建模型
        # 检查索引状态
        if not self.is_built():
            raise RuntimeError("索引尚未构建，请先调用 build_index")

        # 验证参数
        if len(chunks) != len(doc_ids):
            raise ValueError(f"chunks : {len(chunks)} 和 doc_ids : {len(doc_ids)} 的长度不匹配")

        # 拼接 + 重建索引
        all_chunks = self.corpus + list(chunks)
        all_doc_ids = self.doc_ids + list(doc_ids)

        self.build_index(all_chunks, all_doc_ids)



    def remove_by_doc_ids(self, doc_ids_to_remove: List[str]) -> int:
        """按文档 ID 批量删除 chunk（过滤后重建索引）

        Args:
            doc_ids_to_remove: 要删除的文档标识列表

        Returns:
            实际删除的 chunk 数量

        Raises:
            RuntimeError: 如果索引尚未构建
        """
        # 1. 检查索引状态
        if not self.is_built():
            raise RuntimeError("索引尚未构建，请先调用 build_index")

        # 2. 空名单 → 一个都不删，直接返回，避免无意义的重建
        if not doc_ids_to_remove:
            return 0

        # 3. 转 set —— O(1) 查找，否则 list 的 in 是 O(n)，大索引时会很慢
        remove_set = set(doc_ids_to_remove)

        # 4. 过滤：保留 doc_id 不在删除名单里的 chunk
        #    zip(corpus, doc_ids) 把文本和来源一对一捆在一起遍历
        filtered = [
            (chunk, did)
            for chunk, did in zip(self.corpus, self.doc_ids)
            if did not in remove_set
        ]
        # 手动拆回两个列表（Python 不能自动解包 [(a,b), (c,d)] → ([a,c], [b,d])）
        filtered_corpus = [item[0] for item in filtered]
        filtered_doc_ids = [item[1] for item in filtered]

        # 5. 算删除数量（必须在 build_index 之前算，因为 build_index 会更新 self.corpus）
        removed_count = len(self.corpus) - len(filtered_corpus)

        # 6. 边界：所有 chunk 都被删了 → 清空索引，回到初始状态
        #    不能调 build_index([], [])，因为空列表会触发 ValueError
        if not filtered_corpus:
            logger.warning("索引中的所有 chunk 已被删除，索引已清空，可通过 build_index 重新构建")
            self.corpus = []
            self.doc_ids = []
            self._bm25_model = None
            return removed_count

        # 7. 正常情况：用过滤后的数据重建索引
        self.build_index(filtered_corpus, filtered_doc_ids)
        return removed_count



    # ── 状态查询 ──────────────────────────────────────────────────

    def is_built(self) -> bool:
        """检查索引是否已构建

        Returns:
            True 如果已调用 build_index 并成功构建
        """
        # TODO(human): 返回 self._bm25_model 是否为 None
        return self._bm25_model is not None

    def get_stats(self) -> Dict[str, int]:
        """获取索引统计信息

        Returns:
            {"chunk_count": N, "unique_docs": M}
        """
        # TODO(human): 返回 chunk 总数和唯一文档数
        return {
            "chunk_count": len(self.corpus),            # 索引里有多少个 chunk
            "unique_docs": len(set(self.doc_ids)),      # 这些chunk来自多少个不同的文件
        }
