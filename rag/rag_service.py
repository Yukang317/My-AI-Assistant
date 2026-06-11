"""
RAG 服务编排层

将检索管线（BM25 + Vector → RRF → 重排 → 父块回溯）与 LLM 生成串联为完整的
"查询 → 检索 → 生成"流程。

这是路线2（知识检索）的顶层入口，对外暴露两个核心方法：
  - query(): 非流式查询，返回完整结果
  - query_stream(): 流式查询，SSE 异步生成器

架构位置（检索管线全貌）：
  查询 → BM25(关键词) + Vector(语义) 并行 → RRF 文档级融合 → 重排 → 父块回溯 → LLM
   ✅       ✅              ✅                ✅          ⬜     ⬜       ⬜
  (bm25)  (vector_retriever)  (rrf_fusion)   (rag_service)  (rag_service)  (rag_service)

使用方式：
  from rag.rag_service import RagService
  from rag.retrieval.bm25 import BM25Index
  from rag.retrieval.vector_retriever import VectorRetriever    
  from rag.retrieval.rrf_fusion import RRFFusion
  from rag.embedding import EmbeddingService
  from rag.storage.vector_store import MilvusVectorStore

  # 初始化依赖
  emb = EmbeddingService()
  vs = MilvusVectorStore()
  bm25 = BM25Index()
  retriever = VectorRetriever(emb, vs)
  fuser = RRFFusion(k=60)

  # 创建服务
  service = RagService(
      bm25_index=bm25,
      vector_retriever=retriever,
      rrf_fusion=fuser,
  )

  # 非流式查询
  result = service.query("DeepSeek API 怎么调？")
  # → {"answer": "...", "sources": [...], "token_usage": {...}}

  # 流式查询
  async for event in service.query_stream("DeepSeek API 怎么调？"):
      print(event)  # {"type": "sources", ...} / {"type": "delta", ...} / ...
"""

import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import OpenAI, AsyncOpenAI
# from sentence_transformers import CrossEncoder

from rag.config import Config
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.rrf_fusion import RRFFusion

logger = logging.getLogger(__name__)


class RagService:
    """RAG 服务编排层 — 串联检索管线与 LLM 生成。

    职责：
      - 调度 BM25 + Vector 双通道检索
      - 调用 RRF 融合排序
      - 可选的 bge-reranker 重排序
      - 构建 Prompt 并调用 DeepSeek LLM 生成回答
      - 支持流式（SSE）和非流式两种输出模式

    Attributes:
        bm25_index: BM25 关键词索引实例
        vector_retriever: 向量语义检索器实例
        rrf_fusion: RRF 融合排序器实例
        llm_client: OpenAI 兼容的 LLM 客户端（DeepSeek）
        use_rerank: 是否启用重排序（从 Config.USE_RERANK 读取）
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_retriever: VectorRetriever,
        rrf_fusion: RRFFusion,
        llm_client: Optional[OpenAI] = None,  # 只有这个需要兜底策略
    ) -> None:
        """初始化 RAG 服务。

        所有检索组件从外部注入（依赖注入），方便测试时 mock。
        LLM 客户端如果不传，自动根据 Config 创建 DeepSeek 客户端。

        Args:
            bm25_index: 已构建好索引的 BM25Index 实例
            vector_retriever: 已注入 EmbeddingService 和 MilvusVectorStore 的 VectorRetriever
            rrf_fusion: RRFFusion 实例（默认 k=60）
            llm_client: OpenAI 兼容客户端，None 则自动创建
        """
        # TODO(human): 保存四个依赖到 self，从 Config 读取 use_rerank
        self.bm25_index = bm25_index
        self.vector_retriever = vector_retriever
        self.rrf_fusion = rrf_fusion
        self.llm_model = Config.LLM_MODEL
        self.llm_client = llm_client or OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            timeout=60,
        )
        self.llm_async_client = AsyncOpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
            timeout=60,
            max_retries=3,
        )
        
        self.use_rerank = Config.USE_RERANK     # 默认关闭重排序

    # ── 检索管线 ──────────────────────────────────────────────────

    def _retrieve(self, query: str, top_k: int = 10, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行双通道检索 + RRF 融合。

        管线：BM25 关键词检索 + Vector 语义检索 → RRF 融合排序 → top_k 结果。
        两个通道并行执行（当前是顺序调用，但接口设计支持未来并行化）。

        Args:
            query: 用户查询文本
            top_k: 融合后返回的结果数量
            filter_expr: Milvus 标量过滤表达式（仅对向量通道生效），
                        如 'doc_type == "md"'

        Returns:
            RRF 融合后的结果列表，每项包含:
            - doc_id: 文档标识（MinIO object_key）
            - rrf_score: RRF 融合分数
            - bm25_rank / bm25_score: BM25 通道排名和原始分数
            - vector_rank / vector_score: 向量通道排名和原始分数
            - text: 最相关 chunk 文本
            - doc_name / doc_type: 文档名和类型
            - parent_text: 父块完整上下文（从向量结果透传）

        Raises:
            RuntimeError: BM25 索引未构建时抛出
        """
        # 步骤 1：BM25 关键词检索
        # self.bm25_index 就是 __init__ 里注入的 BM25Index 实例
        # 它的 search() 返回 [{"chunk_index": 0, "doc_id": "...", "score": 3.8, "text":"..."}, ...]
        bm25_results = self.bm25_index.search(query, top_k=top_k)

        # 步骤 2：向量语义检索
        # self.vector_retriever 的 search() 多一个 filter_expr 参数
        # 它的返回格式：{"chunk_id": "...", "doc_id": "...", "score": 0.92, "text": "...",
        #                "parent_text": "...", "doc_name": "...", "doc_type": "..."}
        vector_results = self.vector_retriever.search(query, top_k=top_k, filter_expr=filter_expr)
        
        # 步骤 3：RRF 融合——两个通道的结果统一按排名融合
        # fuse() 不关心 BM25 的分数（0~几十）和向量分数（0~1）的尺度差异
        # 它只看"这个文档在 BM25 里排第几、在向量里排第几"
        fused_results = self.rrf_fusion.fuse(bm25_results, vector_results, top_k=top_k)

        return fused_results



    def _rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5,) -> List[Dict[str, Any]]:
        """对检索结果进行重排序。

        使用 bge-reranker-large 模型对每个文档与 query 的相关性重新打分。
        包含安全兜底：向量检索第 1 名强制保留（来自 MilDoc 生产经验）。

        注意：重排模型约 1.3GB，首次加载较慢。可通过 Config.USE_RERANK 关闭。

        Args:
            query: 用户查询文本
            documents: RRF 融合后的文档列表（_retrieve 的输出）
            top_n: 重排后保留的文档数量

        Returns:
            重排后的文档列表，按 rerank_score 降序，最多 top_n 个。
            如果 USE_RERANK 为 False，直接返回前 top_n 个（不做重排）。
        """
        # 如果没开启重排，直接把 RRF 融合结果的前 top_n 个返回
        # self.use_rerank 在 __init__ 里从 Config.USE_RERANK 读取
        if not self.use_rerank:
            return documents[:top_n]
        
        # 1. 加载模型（首次调用时下载 ~1.3GB，后续调用复用）
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("BAAI/bge-reranker-large")

        # 2. 构建 (query, doc_text) 对
        #    每个文档取 text 字段（子块文本，512 tokens，足够判断相关性）
        pairs = [[query, doc["text"]] for doc in documents]

        # 3. 模型预测相关性分数，返回每个 pair 的分数列表
        scores = model.predict(pairs)

        # 4. 把分数注入每个文档的 rerank_score 字段
        for i, score in enumerate(scores):
            documents[i]["rerank_score"] = float(score)

        # 5. 按 rerank_score 降序排列
        reranked = sorted(documents, key = lambda d: d["rerank_score"], reverse=True)

        # 6. 安全兜底：确保向量检索第 1 名不被重排丢掉
        #    向量结果在 documents 中的第 1 个就是向量通道的 top-1
        #    （RRF 融合保持了原始顺序信息，vector_rank=1 的文档就是向量第 1 名）
        if reranked and documents:
            vector_top1 = None
            for doc in documents:
                if doc.get("vector_rank") == 1:
                    vector_top1 = doc
                    break
            if vector_top1 and vector_top1 not in reranked[:top_n]:
                reranked.insert(0, vector_top1)
        
        # 7. 返回前 top_n 个
        return reranked[:top_n]




    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """将检索到的文档列表拼接为 LLM Prompt 中的 context 字符串。

        每个文档格式：
        [来源: {doc_name} ({doc_type})]
        {parent_text 或 text}
        ---

        Args:
            documents: 重排后的最终文档列表

        Returns:
            拼接好的上下文字符串，用于填入 Prompt 模板的 {context} 占位符
        """
        # 如果没有检索到任何文档，返回一个占位提示
        # 这个提示会填入 Prompt 模板的 {context} 位置
        if not documents:
            return "知识库中暂无相关内容。"
        
        # 遍历每个文档，逐个拼成格式化的文本块
        parts = []
        for doc in documents:
            # 优先取 parent_text （父块完整上下文， 2048 tokens）
            # 如果父块不存在（纯 BM25 命中的情况），降级取 text （子块，512 tokens）
            content = doc.get("parent_text") or doc.get("text", "")

            # 取出文档名和类型，给默认值，防止 KeyError
            doc_name = doc.get("doc_name", "未知文档")
            doc_type = doc.get("doc_type", "")

            # 拼成格式化块：
            # [来源：DeepSeek笔记 (md)]
            # ……内容……
            part = f"[来源：{doc_name} ({doc_type})]\n{content}\n---"
            parts.append(part)
        
        # 用两个换行把所有文档块连起来
        return "\n\n".join(parts)


    # ── LLM 生成 ──────────────────────────────────────────────────

    def _build_prompt(self, question: str, context: str) -> str:
        """用 Config.RAG_PROMPT_TEMPLATE 构建最终发送给 LLM 的 Prompt。

        Args:
            question: 用户原始问题
            context: _build_context() 拼接好的上下文字符串

        Returns:
            完整的 Prompt 字符串（包含 system 指令 + context + question）
        """
        # 为什么用 .replace() 而不用 .format()：模板内容（Config.RAG_PROMPT_TEMPLATE）可能包含 Markdown
        # 代码块 ```、JSON 示例 {"key": "value"} 等——这些都有花括号。Python 的 .format() 会把所有 {}
        # 当占位符处理，遇到未转义的就会报 KeyError。.replace() 只替换大括号，避免这个问题。
        prompt = (
            Config.RAG_PROMPT_TEMPLATE
            .replace("{context}", context)
            .replace("{question}", question)
        )
        return prompt



    # ── 公开接口 ──────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 10, filter_expr: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """非流式 RAG 查询：检索 → 重排 → LLM 生成 → 返回完整结果。

        完整的 RAG 管线：
          1. _retrieve(question, top_k, filter_expr)  → BM25 + Vector + RRF
          2. _rerank(question, fused_docs, top_n)     → bge-reranker 精排
          3. _build_context(reranked_docs)            → 拼接 context 字符串
          4. _build_prompt(question, context)         → 组装 Prompt
          5. llm_client.chat.completions.create()     → 调 DeepSeek
          6. 返回 {answer, sources, token_usage, latency_ms}

        Args:
            question: 用户问题
            top_k: 检索阶段返回的候选数量（默认 10）
            filter_expr: Milvus 标量过滤表达式，如 'doc_type == "md"'
            conversation_history: 历史对话 [{"role": "user/assistant", "content": "..."}]
                                  用于多轮对话场景

        Returns:
            {
                "answer": "根据知识库内容...",
                "sources": [
                    {"doc_name": "...", "doc_type": "md", "doc_id": "...", "rerank_score": 0.92},
                    ...
                ],
                "token_usage": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                "latency_ms": 1234,
            }

        Raises:
            RuntimeError: BM25 索引未构建
            ValueError: question 为空
        """
        # 1. 验证 question 非空
        if not question:
            raise ValueError("question 不能为空")
        
        # 2. 记录开始时间（用于计算延迟）
        import time
        start_time = time.time()

        # 3. 检索 + 重排
        # _retrieve() : BM25 + Vector → RRF 融合（top_k=10 个候选）
        fused_docs = self._retrieve(query=question, top_k=top_k, filter_expr=filter_expr)
        # _rerank: 重排序（如果 USE_RERANK=True）或透传前 5 个
        final_docs = self._rerank(query=question, documents=fused_docs, top_n=Config.RERANK_TOP_N)

        # 4. 构建 Prompt
        # _build_context(): 文档列表 → 格式化的上下文字符串
        context = self._build_context(final_docs)
        # _build_prompt: 填充模板 → 完整 Prompt
        prompt = self._build_prompt(question=question, context=context)

        # 步骤 5：调用 LLM
        # messages 是一个列表，每项带 role 和 content
        # 如果有历史对话，把历史消息也加进去（放在当前问题前面）
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": prompt})

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS,
        )

        # 6. 提取 LLM 输出
        answer = response.choices[0].message.content

        # 7. 提取 token 消耗
        token_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # 8. 计算延迟（毫秒）
        latency_ms = int((time.time() - start_time) * 1000)

        # 9. 返回结果
        return {
            "answer": answer,
            "sources": final_docs,          # 来源文档列表
            "token_usage": token_usage,     # token 统计
            "latency_ms": latency_ms,       # 总延迟
            "question": question,           # 原始问题
        }

    async def query_stream(self, question: str, top_k: int = 10, filter_expr: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """流式 RAG 查询（SSE）：先推 sources，再流式推 LLM 生成的 delta。

        与 query() 的区别：
          - 检索和重排阶段相同
          - LLM 调用使用 stream=True
          - 先 yield {"type": "sources", "documents": [...]}
          - 再逐个 yield {"type": "delta", "content": "..."}
          - 最后 yield {"type": "complete", "token_usage": {...}}

        Args:
            question: 用户问题
            top_k: 检索阶段返回的候选数量
            filter_expr: Milvus 标量过滤表达式
            conversation_history: 历史对话

        Yields:
            {"type": "sources", "documents": [...]}   — 先于文本推送
            {"type": "delta", "content": "你好"}       — 逐 token 推送
            {"type": "delta", "content": "，我来"}     — ...
            {"type": "complete", "token_usage": {...}} — 结束信号
            {"type": "error", "content": "..."}        — 出错时推送
        """
        # 步骤 1：验证
        if not question.strip():
            raise ValueError("question 不能为空")

        # 步骤 2：try 包裹全部——出了任何错都 yield error 事件，不让整个流崩溃
        try:
            # 步骤 3：检索 + 重排（和 query() 完全一样）
            fused_docs = self._retrieve(query=question, top_k=top_k, filter_expr=filter_expr)
            final_docs = self._rerank(query=question, documents=fused_docs, top_n=Config.RERANK_TOP_N)

            # 步骤 4：构建 Prompt
            context = self._build_context(final_docs)
            prompt = self._build_prompt(question=question, context=context)
            
            # 步骤 5：先推送 sources 事件
            # 用户第一时间看到“参考了哪些文档”，不用等 LLM 生成完
            yield {"type": "sources", "documents": final_docs}

            # 步骤 6：LLM 流式调用
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            messages.append({"role": "user", "content": prompt})

            # 注意：用的是 AsyncOpenAI 客户端，加了 stream = True
            stream = await self.llm_async_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=Config.LLM_MAX_TOKENS,
                stream=True,  # 开启流式
            )

            # 步骤 7：逐 chunk 推送 delta
            async for chunk in stream:
                # chunk.choices[0].delta.content 是这一小段文本
                # 最后一个 chunk 的 delta.content 可能是 None（只有 finish_reason)
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield {"type": "delta", "content": delta.content}

            # 步骤 8：流结束，推送 complete 事件
            # 注意：流式模式下 response 没有 usage 字段，这里 token_usage 为空
            yield {"type": "complete", "token_usage": {}}

        except Exception as e:
            # 步骤 9：出错，推送 error 事件
            yield {"type": "error", "content": str(e)}



    # ── 状态查询 ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """获取 RAG 服务各组件的统计信息。

        Returns:
            {
                "bm25": {...},
                "vector": {...},
                "rrf": {...},
                "use_rerank": True/False,
                "llm_model": "deepseek-chat",
            }
        """
        return {
            "bm25": self.bm25_index.get_stats(),
            "vector": self.vector_retriever.get_stats(),
            "rrf": self.rrf_fusion.get_stats(),
            "use_rerank": Config.USE_RERANK,
            "llm_model": self.llm_model,
        }

