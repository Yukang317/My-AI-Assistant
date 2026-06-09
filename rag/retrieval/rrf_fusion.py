"""
RRF (Reciprocal Rank Fusion) 融合排序器

将 BM25 和向量检索两个通道的结果合并为统一排序。
RRF 不关心原始分数的绝对值，只关心相对排名，因此天然适合融合异构检索结果。

算法公式：RRF_score(d) = Σ 1/(k + rank_i(d))
  - k: 常数（默认 60），防止高排名项权重过大
  - rank_i(d): 文档 d 在第 i 个检索器中的排名（1-based，即第1名 rank=1）

参考论文：Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and
individual rank learning methods", SIGIR 2009.

融合粒度说明：
  本实现采用**文档级融合**——以 doc_id（MinIO object_key）作为融合键。
  原因：BM25 用 chunk_index（数组位置）标识，Vector 用 chunk_id（业务 ID）标识，
  两者无法直接对应。doc_id 是两个通道共有的、语义一致的标识字段。

  一个文档在单个检索器中可能出现多次（多个 chunk），取最佳排名作为该文档的排名。

使用方式：
  from rag.retrieval.bm25 import BM25Index
  from rag.retrieval.vector_retriever import VectorRetriever
  from rag.retrieval.rrf_fusion import RRFFusion

  fuser = RRFFusion(k=60)
  bm25_results = bm25.search("DeepSeek API", top_k=10)
  vector_results = retriever.search("DeepSeek API", top_k=10)
  fused = fuser.fuse(bm25_results, vector_results, top_k=10)
  # → [{"doc_id": "documents/deepseek.md", "rrf_score": 0.0323,
  #     "bm25_rank": 1, "vector_rank": 3, "text": "...", ...}, ...]
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RRFFusion:
    """RRF 融合排序器 — 将 BM25 和向量检索结果按排名融合。

    不依赖外部服务，纯数学计算。
    k 值越大，高排名和低排名的权重差距越小（越"民主"）。

    Attributes:
        k: RRF 常数，默认 60（学术界标准值）
    """

    def __init__(self, k: int = 60) -> None:
        """初始化 RRF 融合器。

        Args:
            k: RRF 公式中的常数。k=60 是 SIGIR 2009 论文的推荐值。
               越大，排名差异的影响越小；越小，高排名项越占优势。
        """
        # TODO(human): 保存 self.k，加上基本校验（k 必须 > 0，否则除法会崩）
        if k <= 0:
            raise ValueError("k 必须大于 0")

        self.k = k

    # ── 私有方法 ──────────────────────────────────────────────────

    def _build_rank_map(
        self,
        results: List[Dict[str, Any]],
        item_id_field: str,
    ) -> Dict[str, Dict[str, Any]]:
        """从检索结果列表构建 {item_id: {rank, score, text, ...}} 映射。

        前因：BM25/Vector 返回的结果是 chunk 粒度的，同一个文档（如 deepseek.md）的多个 chunk
                会占据多个排名位置。
        函数做的事：从前往后扫，每个文档只取第一次出现（最好排名），跳过后续同文档
                    chunk。
        后果：输出一个 {文档名: 排名} 的查分表，fuse() 拿它套 RRF 公式 1/(k+rank)
                算融合分数——每个文档只算一次，不会被 chunk 数量干扰。

        Args:
            results: 检索器返回的结果列表（已按分数降序排列）
            item_id_field: 用作融合键的字段名，如 "doc_id"

        Returns:
            {
                "doc_id_1": {
                    "rank": 1,        # 1-based 排名
                    "score": 0.95,    # 原始分数
                    "text": "...",    # chunk 文本
                    ...               # 其他字段原样保留
                },
                ...
            }
        """
        # TODO(human): 遍历 results，用 item_id_field 的值做 key，
        # 第一次遇到时记录 rank（enumerate 的 i+1）+ 完整 item 内容
        # 已经见过的跳过（保留最佳排名）
        rank_map = {}

        # 遍历，rank从1开始
        for rank, item in enumerate(results, start=1):
            # 取出融合键的值
            item_id = item.get(item_id_field)

            # 防御：这个 item 没有 doc_id 字段，直接跳过
            if item_id is None:
                continue

            # 去重，已经见过了就跳过，保留第一次（排名最好的那一次）
            # 字典的 in 是 O(1)，deepseek.md 第一次在第 1 名出现时存进去，第 3 名、第 5 名再遇到就直接跳过
            if item_id in rank_map:
                continue
            # 第一次见，存入
            rank_map[item_id] = {
                "rank": rank,  # 注入排名
                # {"rank": rank, **item}：**item 是字典解包。假设 item 是 {"doc_id": "...", "score": 3.8, "text": "..."}，那 {"rank": 1, **item} 就变成 {"rank": 1, "doc_id": "...", "score": 3.8, "text":"..."}。一句话把 rank 注入进去了。
                **item,
            }

        return rank_map

    # ── 核心方法 ──────────────────────────────────────────────────

    def fuse(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """RRF 融合主入口：合并两个检索通道的结果。

        算法步骤：
          1. 分别构建 BM25 和 Vector 的 rank map（以 doc_id 为键）
          2. 收集所有出现过的 doc_id（取并集）
          3. 对每个 doc_id 计算 RRF 分数：
             rrf_score = 1/(k + bm25_rank) + 1/(k + vector_rank)
             （如果某通道没有该 doc_id，对应项为 0）
          4. 按 rrf_score 降序排列，取 top_k

        Args:
            bm25_results: BM25Index.search() 的返回结果
            vector_results: VectorRetriever.search() 的返回结果
            top_k: 融合后返回的结果数量，默认 10

        Returns:
            按 rrf_score 降序排列的结果列表，每项包含:
            - doc_id: str, 文档标识（MinIO object_key）
            - rrf_score: float, RRF 融合分数（越高越相关）
            - bm25_rank: int | None, 在 BM25 结果中的排名
            - bm25_score: float | None, BM25 原始分数
            - vector_rank: int | None, 在向量结果中的排名
            - vector_score: float | None, 向量原始分数
            - text: str, 最相关的 chunk 文本（优先取向量结果）
            - doc_name: str | None, 文档名
            - doc_type: str | None, 文件类型

        Raises:
            ValueError: top_k <= 0 时抛出
        """
        # TODO(human): 实现完整的 RRF 融合流程：
        # 1. 验证 top_k > 0
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        # 2. 调 _build_rank_map 构建两个 rank map（key="doc_id"）
        bm25_rank_map = self._build_rank_map(bm25_results, "doc_id")
        vector_rank_map = self._build_rank_map(vector_results, "doc_id")
        # bm25_rank_map = {
        #       "deepseek.md": {"rank": 1, "score": 3.8, "text": "DeepSeek API 的 base_url..."},
        #       "python.md":   {"rank": 2, "score": 2.1, "text": "Python 异步编程..."},
        #   }

        # vector_rank_map = {
        #       "python.md":   {"rank": 1, "score": 0.92, "text": "Python asyncio 用法...", "doc_name":
        #   "Python笔记", "doc_type": "md"},
        #       "deepseek.md": {"rank": 3, "score": 0.85, "text": "DeepSeek 调用方式...", "doc_name":
        #   "DeepSeek笔记", "doc_type": "md"},
        #       "docker.md":   {"rank": 2, "score": 0.88, "text": "Docker compose...", "doc_name":
        #   "Docker笔记", "doc_type": "md"},
        #   }

        # 3. 取所有 doc_id 的并集
        all_doc_ids = set(bm25_rank_map.keys()) | set(vector_rank_map.keys())

        # 4. 遍历并集，计算每个 doc_id 的 rrf_score
        items = []
        for doc_id in all_doc_ids:
            # 每个文档独立算分，从 0 开始
            rrf_score = 0.0
            if doc_id in bm25_rank_map:
                rrf_score += 1.0 / (self.k + bm25_rank_map[doc_id]["rank"])
            if doc_id in vector_rank_map:
                rrf_score += 1.0 / (self.k + vector_rank_map[doc_id]["rank"])

            # 安全取值：用 .get() 拿整个 item，不存在就是 None
            bm25_info = bm25_rank_map.get(doc_id)
            vector_info = vector_rank_map.get(doc_id)

            # text 优先取 vector（语义匹配的 chunk 通常更完整），没有就取 bm25
            text = (
                vector_info["text"] if vector_info
                else bm25_info["text"] if bm25_info
                else ""
            )
            # doc_name/doc_type 只在 vector 结果里有（BM25 没存这些字段）
            doc_name = vector_info.get("doc_name") if vector_info else None
            doc_type = vector_info.get("doc_type") if vector_info else None

            item = {
                "doc_id": doc_id,
                "rrf_score": rrf_score,
                "bm25_rank": bm25_info["rank"] if bm25_info else None,
                "bm25_score": bm25_info["score"] if bm25_info else None,
                "vector_rank": vector_info["rank"] if vector_info else None,
                "vector_score": vector_info["score"] if vector_info else None,
                "text": text,
                "doc_name": doc_name,
                "doc_type": doc_type,
            }
            items.append(item)

        # 5. 按 rrf_score 降序排列
        items.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 6. 返回 top_k
        return items[:top_k]
        

    # ── 状态查询 ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取融合器配置信息。

        Returns:
            {"k": 60, "algorithm": "RRF (Reciprocal Rank Fusion)"}
        """
        # TODO(human): 返回 self.k 和算法名
        return {
            "k": self.k,
            "algorithm": "RRF (Reciprocal Rank Fusion)",
        }



# 两个通道输入 fuse 的字段

#   BM25 (bm25.py search 返回)：
#   {
#       "chunk_index": 3,           # chunk 在 corpus 中的位置
#       "doc_id": "documents/xxx",  # MinIO 文件路径
#       "score": 3.8,               # BM25 TF-IDF 分数（0~几十）
#       "text": "DeepSeek API...",  # chunk 原文
#   }
  
#   Vector (vector_retriever.py search 返回)：
#   {
#       "chunk_id": "报告_A_p0_c3",  # 子块业务 ID
#       "doc_id": "documents/xxx",  # MinIO 文件路径
#       "score": 0.92,              # 内积相似度（0~1）
#       "text": "子块内容...",       # 子块原文（512 tokens）
#       "parent_text": "...",       # 父块原文（2048 tokens）
#       "doc_name": "DeepSeek笔记",  # 文档名
#       "doc_type": "md",           # 文件类型
#   }
  
#   fuse 输出字段

#   {
#       "doc_id":     "documents/xxx",   # ← 融合键，两个通道都有
#       "rrf_score":  0.0323,           # ← 新增：RRF 公式算出来的
#       "bm25_rank":  1,                # ← 从 BM25 rank_map 来
#       "bm25_score": 3.8,              # ← 从 BM25 rank_map 来
#       "vector_rank":  3,              # ← 从 Vector rank_map 来
#       "vector_score": 0.85,           # ← 从 Vector rank_map 来
#       "text":       "...",            # ← 优先 Vector，没有才 BM25
#       "doc_name":   "DeepSeek笔记",    # ← 只有 Vector 有
#       "doc_type":   "md",             # ← 只有 Vector 有
#   }

#   为什么这样设计？

#   ┌──────────────────────────┬─────────┬─────────────────────────────────────────────────────┐
#   │           字段           │  来自   │                    为什么选这个                     │
#   ├──────────────────────────┼─────────┼─────────────────────────────────────────────────────┤
#   │ bm25_rank/score +        │ 双方    │ 保留原始排名和分数，方便排查"为什么搜到这个"        │
#   │ vector_rank/score        │         │                                                     │
#   ├──────────────────────────┼─────────┼─────────────────────────────────────────────────────┤
#   │ text                     │ Vector  │ 语义匹配的 chunk 通常更完整；BM25                   │
#   │                          │ 优先    │ 可能因为关键词碰巧命中一个碎片                      │
#   ├──────────────────────────┼─────────┼─────────────────────────────────────────────────────┤
#   │ doc_name/doc_type        │ 只取    │ BM25 根本没存这两个字段（它只存了 corpus +          │
#   │                          │ Vector  │ doc_ids，没存元信息）                               │
#   └──────────────────────────┴─────────┴─────────────────────────────────────────────────────┘

#   一句话：输出保留了双方原始分数用于"可解释性"，文本优先取 Vector 的（更完整），元信息只能从
#   Vector 拿。