"""
网页搜索工具 — Exa 神经搜索 + Tavily AI 搜索，两种引擎可供选择。

灵感引擎的核心依赖之一。跨领域知识的获取不依赖 RAG（RAG 只存个人文档），
而是通过 (1) 激活大模型预训练参数 (2) 联网搜索 (3) 特定外部资料库。
本模块负责渠道 (2)，提供两种互补的搜索模式。

模式对比（供 LLM 选工具时参考）：
┌──────────┬────────────────────────┬──────────────────────────┐
│          │ Exa 神经搜索            │ Tavily AI 搜索            │
├──────────┼────────────────────────┼──────────────────────────┤
│ 搜索方式 │ Embedding 语义匹配      │ 关键词 + AI 排序          │
│ 擅长场景 │ 跨领域概念关联、学术    │ 实时信息、事实核查、新闻  │
│ 结果特点 │ 用词不同但概念相关      │ 结构化、带相关性评分      │
│ 时效性   │ 一般（偏长期内容）      │ 强（实时索引）            │
│ 中文支持 │ 可（建议中英混合）      │ 良好                      │
│ 价格     │ 按次（有免费额度）      │ 按次（有免费额度）        │
└──────────┴────────────────────────┴──────────────────────────┘
"""
import logging
from agent.tools.base import ToolContext, ToolResult

logger = logging.getLogger(__name__)

# ── API Key（从 .env / 环境变量读取） ──────────────────────────────
# 两种引擎独立配置：可以只配一个（另一个被调用时返回密钥缺失提示），也可以两个都配
_EXA_API_KEY: str = ""
_TAVILY_API_KEY: str = ""


def _load_keys() -> None:
    """惰性加载 API Key，避免 import 时读环境变量（方便单测 mock）"""
    import os

    global _EXA_API_KEY, _TAVILY_API_KEY
    if not _EXA_API_KEY:
        _EXA_API_KEY = os.getenv("EXA_API_KEY", "")
    if not _TAVILY_API_KEY:
        _TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════
# 内部搜索实现
# ══════════════════════════════════════════════════════════════════════


def _search_exa(query: str, max_results: int = 5) -> list[dict]:
    """Exa 神经语义搜索。

    通过 embeddings 做语义匹配：输入"分布式计算"，能命中"MapReduce""边缘节点"等
    用词不同但概念相关的内容。这正是灵感引擎 Phase 1 跨领域发散的核心能力。

    Args:
        query: 搜索查询（支持中英文，建议中英混合以获得更好覆盖）
        max_results: 最多返回结果数，默认 5

    Returns:
        [{"title": str, "url": str, "snippet": str}, ...]
        snippet 截取前 500 字符
    """
    from exa_py import Exa

    exa = Exa(api_key=_EXA_API_KEY)
    result = exa.search_and_contents(
        query,
        num_results=max_results,
        text=True,                  # 要求返回页面正文摘要（否则只有标题+URL）
    )

    results = []
    for r in result.results:
        text = r.text or ""
        results.append({
            "title": r.title or "",
            "url": r.url or "",
            "snippet": text[:500],  # 截断过长的摘要，LLM 上下文窗口有限
        })

    return results


def _search_tavily(query: str, max_results: int = 5) -> list[dict]:
    """Tavily AI 搜索。

    Tavily 专为 AI Agent 流水线设计：返回结果已经过结构化清洗，
    包含标题、URL、内容摘要和相关性评分，LLM 可直接消费。

    Args:
        query: 搜索查询（支持中英文）
        max_results: 最多返回结果数，默认 5

    Returns:
        [{"title": str, "url": str, "snippet": str, "score": float}, ...]
        score 为 Tavily 内置的相关性评分（0-1），非 LLM 主观评分
    """
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=_TAVILY_API_KEY)
    response = tavily.search(
        query=query,
        max_results=max_results,
        search_depth="basic",       # "basic" = 快速（~1s），"advanced" = 深度（~3s，更全）
    )

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "score": r.get("score", 0.0),
        })

    return results


# ══════════════════════════════════════════════════════════════════════
# 工具函数（公开给 Agent 调用）
# ══════════════════════════════════════════════════════════════════════


def exa_search(ctx: ToolContext, query: str, max_results: int = 5) -> ToolResult:
    """Exa 神经语义搜索：擅长概念理解和跨领域关联。

    使用 embeddings 做神经网络级别的语义匹配（而非关键词），擅长找到：
    - 与查询概念相关但用词不同的内容（「替代能源」→「核聚变研究」）
    - 跨领域类比和关联案例
    - 学术论文、技术文档等长尾内容

    需要 EXA_API_KEY 环境变量。免费额度注册：https://exa.ai

    Args:
        ctx: 工具调用上下文（含 session_id, user_id）
        query: 搜索查询字符串（支持中英文）
        max_results: 最多返回结果数，默认 5

    Returns:
        ToolResult: 成功时 data 为格式化摘要，artifacts={"results": [...]}
    """
    _load_keys()

    if not _EXA_API_KEY:
        return ToolResult(
            success=False,
            data="",
            error="未配置 EXA_API_KEY，请在 .env 中设置。免费注册: https://exa.ai",
        )

    try:
        results = _search_exa(query, max_results=max_results)

        if not results:
            return ToolResult(
                success=False,
                data="",
                error=f"Exa 未找到关于 '{query}' 的搜索结果，建议换 Tavily 试试",
            )

        summary = _format_results("Exa 神经搜索", results)
        return ToolResult(
            success=True,
            data=summary,
            artifacts={"results": results, "provider": "exa"},
        )

    except Exception as e:
        logger.error(f"Exa 搜索失败: {e}")
        return ToolResult(
            success=False,
            data="",
            error=f"Exa 搜索异常: {str(e)}",
        )


def tavily_search(ctx: ToolContext, query: str, max_results: int = 5) -> ToolResult:
    """Tavily AI 搜索：专为 AI Agent 设计的实时网页搜索。

    返回结构化结果（标题/URL/内容/相关性评分），适合：
    - 获取最新事实、新闻动态、实时数据
    - 需要高精度结果的事实核查类查询
    - 时间敏感型问题（「今天发生了什么」）

    需要 TAVILY_API_KEY 环境变量。免费额度注册：https://tavily.com

    Args:
        ctx: 工具调用上下文（含 session_id, user_id）
        query: 搜索查询字符串（支持中英文）
        max_results: 最多返回结果数，默认 5

    Returns:
        ToolResult: 成功时 data 为格式化摘要，artifacts={"results": [...]}
    """
    _load_keys()

    if not _TAVILY_API_KEY:
        return ToolResult(
            success=False,
            data="",
            error="未配置 TAVILY_API_KEY，请在 .env 中设置。免费注册: https://tavily.com",
        )

    try:
        results = _search_tavily(query, max_results=max_results)

        if not results:
            return ToolResult(
                success=False,
                data="",
                error=f"Tavily 未找到关于 '{query}' 的搜索结果，建议换 Exa 试试",
            )

        summary = _format_results("Tavily AI 搜索", results)
        return ToolResult(
            success=True,
            data=summary,
            artifacts={"results": results, "provider": "tavily"},
        )

    except Exception as e:
        logger.error(f"Tavily 搜索失败: {e}")
        return ToolResult(
            success=False,
            data="",
            error=f"Tavily 搜索异常: {str(e)}",
        )


# ══════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════


def _format_results(source: str, results: list[dict]) -> str:
    """将原始搜索结果格式化为 LLM 可直接阅读的文本。

    格式：
        【{来源名称}】共 N 条结果：
        [1] 标题
            URL: url
            摘要: 内容

    Args:
        source: 来源名称（如 "Exa 神经搜索"）
        results: 原始结果列表

    Returns:
        格式化后的多行文本字符串
    """
    parts = [f"【{source}】共 {len(results)} 条结果：\n"]
    for i, r in enumerate(results, 1):
        # score 是 Tavily 特有的字段，Exa 没有，所以用 .get
        score_suffix = f" (相关性: {r['score']:.2f})" if r.get("score") else ""
        parts.append(
            f"[{i}] {r['title']}{score_suffix}\n"
            f"    URL: {r['url']}\n"
            f"    摘要: {r['snippet']}\n"
        )
    return "\n".join(parts)
