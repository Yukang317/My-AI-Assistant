"""
Agent 状态定义 — MainState TypedDict + StateField 枚举

参考 sagt_agent 的 State 设计模式：
- InputState (用户输入) → IntermediateState (执行中间态) → OutputState (最终输出)
- 用 StateField 枚举替代裸字符串 key，保证编译期安全

新增 TRACE_ID / trace_id，供 Graph 节点在一次 invoke 内传递追踪 ID。
"""

from typing import Annotated, Any, Optional, TypedDict, NotRequired
# 它和 Optional[str]（即 str | None）是两回事：Optional 表示 key 一定存在但值可能是 None；NotRequired 表示 key 压根可以不出现
# - 在这个项目里用它很合理：trace_id 只在 run_graph()
#   入口写入一次，load_context、intent_route 等下游节点只读不写，且旧代码 /
#   测试可能完全不给 trace_id，用 NotRequired 确保了向后兼容

from operator import add
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

###############################################################################
# StateField 枚举 — 所有 State key 集中管理
###############################################################################


class StateField:
    """主图 State 字段名枚举，避免裸字符串分散在各处。"""
    # 用户输入
    SESSION_ID: str = "session_id"
    USER_QUESTION: str = "user_question"

    # 追踪 ID
    TRACE_ID: str = "trace_id"

    # 上下文
    HISTORY_MESSAGES: str = "history_messages"
    MEMORY_CONTEXT: str = "memory_context"
    # 路由
    INTENT: str = "intent"
    TARGET_TOOL: str = "target_tool"
    # 工具执行
    TOOL_CALLS: str = "tool_calls"
    TOOL_RESULTS: str = "tool_results"
    # 循环控制
    TURN_COUNT: str = "turn_count"
    NEED_CONTINUE: str = "need_continue"
    # 最终输出
    FINAL_RESPONSE: str = "final_response"


    # 循环控制（阶段 6.2 while-true）
    MAX_TURNS: str = "max_turns"
    LOOP_STOP_REASON: str = "loop_stop_reason"
    TOOL_BUDGET_USED: str = "tool_budget_used"
    TIME_BUDGET_MS: str = "time_budget_ms"
    ELAPSED_MS: str = "elapsed_ms"
    TOKEN_BUDGET_ESTIMATE: str = "token_budget_estimate"
    TOKEN_BUDGET_USED_ESTIMATE: str = "token_budget_used_estimate"

    # 灵感引擎（阶段 6.5）
    INSPIRE_QUERIES: str = "inspire_queries"
    INSPIRE_EVIDENCE: str = "inspire_evidence"  # 并行搜索收集的 ToolResult 列表


###############################################################################
# MainState — LangGraph 主状态 TypedDict
###############################################################################

class MainState(TypedDict):
    """LangGraph 主图 State schema，定义所有节点共享的状态字段。

    字段分五层：
    - 用户输入: session_id, user_question
    - 上下文: history_messages, memory_context (load_context 节点写入)
    - 路由结果: intent, target_tool (intent_route 节点写入)
    - 工具执行: tool_results (Annotated + add reducer，追加而非覆盖)
    - 循环控制: turn_count, need_continue (阶段 6.2 启用)
    - 最终输出: final_response (result_synthesis 节点写入)
    """
    # ── 用户输入 ──
    session_id: str              # 会话 ID，用作 thread_id 实现多会话隔离
    user_question: str           # 用户当前提问原文

    # H2 追踪（可选，仅 invoke 入口写入）
    trace_id: NotRequired[str]

    # ── 上下文 ──
    history_messages: Annotated[list[BaseMessage], add_messages]    # 最近 20 条历史消息（load_context 节点写入）
    memory_context: str          # MEMORY.md 文件内容，用户画像与偏好

    # ── 路由结果 ──
    intent: str                  # LLM 判断的用户意图（general_chat / use_tool / unclear）
    target_tool: str | None      # 目标工具名，对应 TOOLS 注册表的 key，无工具时为 None

    # ── 工具执行 ──
    tool_results: Annotated[list, add]  # 工具调用结果列表，add reducer 追加而非覆盖

    # ── 循环控制 ──
    turn_count: int              # 当前 while-true 循环次数（阶段 6.1 用不上，6.2 启用）
    need_continue: bool          # 是否需要继续循环调用工具

    # ── 最终输出 ──
    final_response: str          # result_synthesis 生成的最终回复文本


    # 循环控制（阶段 6.2 while-true，均可选，invoke 入口可覆盖默认值）
    max_turns: NotRequired[int]              # 最大循环次数
    loop_stop_reason: NotRequired[str]       # 停止原因枚举，如 reach_max_turns / llm_stop
    tool_budget_used: NotRequired[int]       # 已用工具调用次数
    time_budget_ms: NotRequired[int]         # 总超时预算（毫秒）
    elapsed_ms: NotRequired[int]             # 本节点累计耗时（毫秒）
    token_budget_estimate: NotRequired[int]  # token 软预算上限（字符数代理）
    token_budget_used_estimate: NotRequired[int]  # 节点结束时累计估算值

    # 灵感引擎（阶段 6.5）
    inspire_queries: NotRequired[list[str]]  # 发散阶段生成的查询列表
    inspire_evidence: NotRequired[list]      # 告诉检查器只是个列表，给人看的是：list[ToolResult], TypedDict 不做泛型

