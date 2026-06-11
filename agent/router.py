"""
意图路由 — LLM 分析用户输入，判断意图并选择对应工具

借鉴 sagt_agent 的 intent_detection 节点设计：
- 用 LangChain ChatModel 做意图识别
- 返回目标工具名 + 路由键
- 不强制依赖路由准确性（有 fallback 兜底）
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel


# ── 路由结果 ──────────────────────────────────────────────────

# TODO(human): 定义 route_result 数据类或 TypedDict
# 说明：
#   1. 至少包含 intent (str) 和 target_tool (Optional[str]) 两个字段
#   2. intent 的可能值：general_chat, use_tool, unclear
#   3. target_tool 是工具注册表中 TOOLS 的 key，intent 为 use_tool 时才非空


# ── System Prompt ─────────────────────────────────────────────

# TODO(human): 定义 ROUTER_SYSTEM_PROMPT: str 常量
# 说明：
#   1. 提示词要求 LLM 分析用户意图并判断是否需要调用工具
#   2. 如果用工具，返回目标工具名（从 list_tools() 获取可用工具列表）
#   3. 如果不需要工具（普通闲聊），返回 general_chat
#   4. 如果意图不清晰，返回 unclear
#   5. 输出格式：JSON {"intent": "...", "target_tool": "..." 或 null}


# ── 路由函数 ──────────────────────────────────────────────────

# TODO(human): 实现 route_intent(question: str, model: BaseChatModel, available_tools: list[dict]) -> route_result
# 说明：
#   1. 接收用户问题 + LangChain ChatModel + 可用工具列表
#   2. 把可用工具列表（name + description）注入 ROUTER_SYSTEM_PROMPT
#   3. 调用 model.invoke() 获取 LLM 响应
#   4. 解析 LLM 返回的 JSON，提取 intent 和 target_tool
#   5. 用 langchain_core.messages.HumanMessage 和 SystemMessage 构造消息
#   6. JSON 解析失败时，fallback 到 general_chat（不因路由失败而中断整个流程）


# TODO(human): 实现 get_route_key(target_tool: Optional[str]) -> str 函数
# 说明：
#   1. LangGraph 条件边需要一个返回字符串的函数
#   2. target_tool 非空时返回 "use_tool"，否则返回 "general_chat"
#   3. 这个返回值对应 main_graph.py 中条件边的目标节点名
