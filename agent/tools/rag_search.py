"""
RAG 搜索工具 — 把现有 RAG 检索能力封装为标准工具

这是阶段 6.1 的示例工具，验证工具注册 → 调用 → 结果合成的完整链路。
后续新增工具都参照这个模式。
"""
import logging
from agent.tools.base import ToolContext, ToolResult
from step23_rag import get_rag_service

# ── 工具函数 ──────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def rag_search(ctx: ToolContext, query: str) -> ToolResult:
    """基于 RAG 知识库搜索文档内容，返回 LLM 生成的回答和引用来源。
    
    这是 Agent 工具层对现有 RAG 检索能力的薄封装：
    step23_rag.get_rag_service() → RagService.query() → ToolResult
    
    Args:
        ctx: 工具调用上下文（含 session_id, user_id）
        query: 用户原始查询字符串
    
    Returns:
        ToolResult: 成功时 data=LLM回答, artifacts={"sources": [...]}
    """
    try:
        # 获取 RAG 服务单例（已在 step23_rag 启动时初始化）
        service = get_rag_service()          # RagService 实例，内部持有 BM25 + 向量 + RRF + 重排
        result = service.query(              # 调用检索+生成管线
            question=query,                  # 注意参数名是 question 不是 query
            top_k=5,                         # 只返回 top5 文档片段给 LLM
        )
        
        answer = result.get("answer", "")    # LLM 基于检索结果生成的回答
        sources = result.get("sources", [])  # 引用的文档片段列表
        
        if answer:
            return ToolResult(
                success=True,
                data=answer,
                artifacts={"sources": sources},
            )
        else:
            return ToolResult(
                success=False,
                data="",
                error="RAG 检索未找到相关内容",
            )
    
    except Exception as e:
        logger.error(f"RAG 搜索失败: {e}")
        return ToolResult(
            success=False,
            data="",
            error=str(e),
        )
