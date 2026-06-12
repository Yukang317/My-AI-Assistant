"""
Agent 状态定义 — MainState TypedDict + StateField 枚举

参考 sagt_agent 的 State 设计模式：
- InputState (用户输入) → IntermediateState (执行中间态) → OutputState (最终输出)
- 用 StateField 枚举替代裸字符串 key，保证编译期安全
"""

from typing import Annotated, Any, Optional, TypedDict
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

    # ── 上下文 ──
    history_messages: list       # 最近 20 条历史消息（load_context 节点写入）
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
