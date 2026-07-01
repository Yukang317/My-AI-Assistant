"""
灵感引擎 - Phase 1 发散检索

职责：
  用户问题 → 轻量 LLM 生成多角度查询 → 并行 RAG+Exa/Tavily 搜索 → 写入 State

数据流位置：
  intent_route(inspire) → 【本节点】 → inspire_converge
"""

# 让所有注解（list[str] 之类）延迟求值，避免循环引用
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed     # 并行搜索

from langchain_core.language_models import BaseChatModel

from agent.bound import check_bound                         # 安全检查
from agent.state import MainState, StateField
from agent.tools.base import ToolContext, ToolResult        # 工具上下文和结果
from agent.tools.registry import get_tool, get_tool_bound   # 工具注册表
from agent.trace import trace_inspire_diverge               # 追踪打点

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────────

# Phase 1 ：最多 3 条查询，控制 token 和 API 成本
DEFAULT_MAX_QUERIES = 3

# 每条 query 并行搜索 2 个工具：个人文档 + 跨域外网
DIVERGE_SEARCH_TOOLS = ("rag_search", "exa_search")

# main_graph.py 里已有同样的映射，这里复制一份避免循环 import
BOUND_ACTION_MAP: dict[str, str] = {
    "READ_ONLY": "读取数据",
    "NETWORK": "网络搜索",
    "WRITE": "写入数据",
}

# 轻量 LLM prompt：只输出 JSON 查询列表
DIVERGE_QUERY_PROMPT = """你是一位**跨领域研究员**（Cross-domain Researcher）——类似 MIT Media Lab 或 IDEO 的研究方法论专家。
你的专长是把一个问题从当前领域「翻译」到其他学科领域去寻找类比和启发，而不是在单一领域里找精确答案。

## 用户画像（辅助理解用户背景）
{memory_context}

## 任务
根据用户问题，生成 {max_queries} 条**不同角度**的搜索查询，用于并行检索。
角度示例：同域深入 / 跨域类比 / 反直觉视角 / 实践案例 / 学术理论

## 输出规则
1. 严格输出 JSON，不要任何其他文字
2. 格式：{{"queries": ["查询1", "查询2", "查询3"]}}
3. 每条查询 10~30 字，中英文混合效果更好
4. 不要重复相同角度

## 用户问题
{question}
"""

# —— 1. JSON 解析（容错） ──────────────────────────────────────
def _parse_queries_json(raw: str, fallback_question: str, max_queries: int) -> list[str]:
    """解析 LLM 返回的查询 JSON，失败时用原问题兜底。

    Args:
        raw: LLM 原始输出（可能含 ```json 包裹）
        fallback_question: 解析失败时的兜底查询
        max_queries: 最多保留几条

    Returns:
        查询字符串列表，长度 1~max_queries
    """
    text = raw.strip()

    # 剥掉 ```json ... ``` 包裹（LLM 常见输出格式）
    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # 尝试 JSON 解析
    try:
        data = json.loads(text)
        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise ValueError("queries 不是列表")

        # 过滤空字符串，截断到 max_queries
        cleaned = [str(q).strip() for q in queries if str(q).strip()]
        if cleaned:
            return cleaned[:max_queries]

    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning("查询 JSON 解析失败，使用兜底: %s", e)

    # 兜底：至少用原问题搜一次
    return [fallback_question]


# ── 2. 轻量 LLM 生成查询 ──────────────────────────────────────

def generate_diverge_queries(
    question: str,
    memory_context: str,
    model: BaseChatModel,
    max_queries: int = DEFAULT_MAX_QUERIES,
) -> list[str]:
    """调用 LLM 生成多角度搜索查询列表。

    Args:
        question: 用户原始问题
        memory_context: MEMORY.md 内容，帮助 LLM 理解用户背景
        model: LangChain ChatModel 实例
        max_queries: 最多生成几条查询

    Returns:
        查询字符串列表
    """
    prompt = DIVERGE_QUERY_PROMPT.format(
        question=question,
        memory_context=memory_context.strip() or "（暂无）",
        max_queries=max_queries,
    )

    try:
        response = model.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw if isinstance(raw, str) else str(raw)
    except Exception as e:
        logger.error("查询生成 LLM 调用失败: %s", e)
        return [question]

    # 解析 JSON 查询列表
    return _parse_queries_json(raw, fallback_question=question, max_queries=max_queries)


# ── 3. 单次搜索（含 BOUND 安检） ───────────────────────────────

def _run_single_search(
    tool_name: str,
    query: str,
    ctx: ToolContext,
) -> ToolResult:
    """执行单个工具的单次搜索，先过 BOUND 安检。

    复用 main_graph.tool_execute 的安检模式：
    get_tool_bound → BOUND_ACTION_MAP → check_bound → 执行工具

    Args:
        tool_name: 工具注册表 key，如 "rag_search"
        query: 搜索查询字符串
        ctx: ToolContext，含 session_id

    Returns:
        ToolResult，失败时 success=False
    """
    try:
        bound_category = get_tool_bound(tool_name)  # 获取工具的 BOUND 分类，是纯字符串，如 "READ_ONLY"、"NETWORK"、"WRITE"
        action_desc = BOUND_ACTION_MAP.get(bound_category, "未知操作")  # 获取 action 描述
        action = f"调用工具:{tool_name} {action_desc}"  # 构建 action 描述
        allowed, reason = check_bound(action, query)

        if not allowed:
            return ToolResult(success=False, data="", error=f"BOUND 拒绝: {reason}")

        tool_func = get_tool(tool_name)
        return tool_func(ctx, query)  # 执行工具

    except Exception as e:
        return ToolResult(success=False, data="", error=str(e))


# ── 4. 并行搜索编排 ───────────────────────────────────────────

def execute_parallel_searches(
    queries: list[str],
    session_id: str,
    tools: tuple[str, ...] = DIVERGE_SEARCH_TOOLS,
    max_workers: int = 6,
) -> list[ToolResult]:
    """对每条 query × 每个 tool 并行执行搜索。
    
    3 条 query × 2 工具 = 6 个任务，用 ThreadPoolExecutor 并行。
    工具函数是同步阻塞的（HTTP 调用），线程池合适。

    Args:
        queries: 发散查询列表
        session_id: 会话 ID，传给 ToolContext
        tools: 要并行调用的工具名元组
        max_workers: 线程池大小，默认 6

    Returns:
        全部 ToolResult 列表（含失败的，converge 阶段会过滤）
    """
    if not queries:
        return []

    ctx = ToolContext(session_id=session_id)    # 包含当前会话 ID 和用户 ID，工具可以据此访问数据库、读取用户偏好等。
    results: list[ToolResult] = []

    # 1. 构造 (tool_name, query) ，任务列表
    # 例：queries=["认知科学启发", "跨界类比"], tools=("rag_search", "exa_search")
    # → tasks = [("rag_search","认知科学启发"), ("exa_search","认知科学启发"),
    #            ("rag_search","跨界类比"),   ("exa_search","跨界类比")]
    tasks: list[tuple[str, str]] = []
    for query in queries:
        for tool_name in tools:
            tasks.append((tool_name, query))

    # 并行执行
    # 2. 开线程池，提交所有任务
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            # 提交任务
            # 遍历任务列表，提交任务到线程池，返回一个 Future 对象（("rag_search", "认知科学")）
            executor.submit(_run_single_search, tool_name, query, ctx): (tool_name, query)
            for tool_name, query in tasks
        }
#       future_map = {
#       <Future at 0x7f8a1c001234 state=running>:  ("rag_search", "认知科学 
#       记忆提取 RAG系统启发"),
#       ...
#       }

        # 3. as_completed 按完成顺序遍历处理完成的结果
        for future in as_completed(future_map):
            # 获取这个 future 对应的 (tool_name, query)
            tool_name, query = future_map[future]
            try:
                # 阻塞等待这个 future 的结果
                result = future.result()
                # result = ToolResult(success=True, data="...", error=None, artifacts={"tool": "rag_search", "query": "认知科学 记忆提取 RAG系统启发"})

                # 在 artifacts 里标记来源，converge 格式化 evidence 时用
                if result.artifacts is None:
                    result.artifacts = {}
                result.artifacts["tool"] = tool_name
                result.artifacts["query"] = query
                results.append(result)
            except Exception as e:
                # 单个搜索失败不影响整体，记录一条失败的 ToolResult
                results.append(ToolResult(
                    success=False,
                    data="",
                    error=str(e),
                    artifacts={"tool": tool_name, "query": query},
                ))

    return results


# ── 5. LangGraph 节点入口 ───────────────────────────────────────────

def inspire_diverge(state: MainState, model: BaseChatModel) -> dict:
    """灵感引擎发散节点：生成查询 + 并行搜索 + 写入 State。

    LangGraph 节点签名：(state) -> dict。
    main_graph.py 的适配层负责注入 model。

    读取: user_question, memory_context, session_id, trace_id
    写入: inspire_queries, inspire_evidence

    Args:
        state: 当前 MainState
        model: LangChain ChatModel 实例

    Returns:
        dict: State 更新字段
    """
    question = state.get(StateField.USER_QUESTION, "")
    memory_context = state.get(StateField.MEMORY_CONTEXT, "")
    session_id = state.get(StateField.SESSION_ID, "")
    trace_id = state.get(StateField.TRACE_ID, "")

    t0 = time.monotonic()

    # ① 轻量 LLM 生成多角度查询。不访问用户数据所以没加session id
    queries = generate_diverge_queries(
        question=question,
        memory_context=memory_context,
        model=model,
    )

    # ② 并行搜索
    evidence = execute_parallel_searches(
        queries=queries,
        session_id=session_id,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ③ H2 追踪打点
    if trace_id:
        trace_inspire_diverge(
            trace_id,
            session_id,
            queries=queries,
            evidence_count=len(evidence),
            latency_ms=elapsed_ms,
        )

    # ④ 返回 State 更新
    return {
        StateField.INSPIRE_QUERIES: queries,
        StateField.INSPIRE_EVIDENCE: evidence,
    }