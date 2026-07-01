"""
RAG 搜索工具 — 把现有 RAG 检索能力封装为标准工具

这是阶段 6.1 的示例工具，验证工具注册 → 调用 → 结果合成的完整链路。
后续新增工具都参照这个模式。
"""
import logging
from agent.tools.base import ToolContext, ToolResult

# ── 工具函数 ──────────────────────────────────────────────────

logger = logging.getLogger(__name__)


def rag_search(ctx: ToolContext, query: str) -> ToolResult:
    """基于 RAG 知识库搜索文档内容，返回 LLM 生成的回答和引用来源。
    
    这是 Agent 工具层对现有 RAG 检索能力的薄封装：
    app.get_rag_service() → RagService.query() → ToolResult
    
    Args:
        ctx: 工具调用上下文（含 session_id, user_id）
        query: 用户原始查询字符串
    
    Returns:
        ToolResult: 成功时 data=LLM回答, artifacts={"sources": [...]}
    """
    try:
        # 惰性导入：避免 app ↔ agent.tools 循环导入
        from app import get_rag_service
        service = get_rag_service()          # RagService 实例，内部持有 BM25 + 向量 + RRF + 重排
        result = service.query(              # 调用检索+生成管线
            question=query,                  # 注意参数名是 question 不是 query
            top_k=5,                         # 只返回 top5 文档片段给 LLM
        )
        
        answer = result.get("answer", "")    # LLM 基于检索结果生成的回答
        sources = result.get("sources", [])  # 引用的文档片段列表
        
        # RagService 检索为空时会带上 no_result 标记并跳过 LLM；
        # 没有该标记的旧路径则用 sources 是否为空兜底判断。
        no_result = result.get("no_result", False) or not sources

        # 空结果：返回 success=True + 诚实文案（而非 success=False 的"搜索失败"）。
        # 这样 result_synthesis 会把"知识库没有相关内容"作为依据如实转达用户，
        # 不会被包装成系统错误，也不让合成 LLM 拿空结果编造。
        if no_result:
            return ToolResult(
                success=True,
                data="知识库中没有检索到与该问题相关的内容。",
                artifacts={"sources": []},
            )

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
