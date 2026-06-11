"""
RAG 搜索工具 — 把现有 RAG 检索能力封装为标准工具

这是阶段 6.1 的示例工具，验证工具注册 → 调用 → 结果合成的完整链路。
后续新增工具都参照这个模式。
"""

from agent.tools.base import ToolContext, ToolResult


# ── 工具函数 ──────────────────────────────────────────────────

# TODO(human): 实现 rag_search(ctx: ToolContext, query: str) -> ToolResult 函数
# 说明：
#   1. 接收 ToolContext 和用户查询字符串
#   2. 导入并调用 rag.rag_service.query(query=query, use_rag=True, top_k=5)
#      - rag.rag_service 是现有的 RAG 编排层（已实现并运行正常）
#      - query() 方法返回 dict，包含 "answer" + "sources" 两个 key
#      - answer: LLM 基于检索结果生成的回答文本
#      - sources: 引用的文档片段列表
#   3. 如果 answer 非空，构造成功的 ToolResult：
#      - success=True
#      - data=answer（LLM 基于检索结果生成的回答）
#      - artifacts={"sources": sources}（引用来源）
#   4. 如果 answer 为空或出现异常，构造失败的 ToolResult
#   5. 异常处理：try/except 包裹，捕获所有异常并返回 ToolResult(success=False, error=str(e))
#   6. 导入：
#      - from rag.rag_service import query
#      - 或 from rag import rag_service（然后用 rag_service.query()）
#
#   提示：这个工具和 step23_rag.py 中 /chat 端点的 RAG 检索逻辑是一样的，
#   只不过这里多了一层 ToolContext/ToolResult 的包装。
