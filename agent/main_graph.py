"""
主图构建 — LangGraph StateGraph 组装

架构：
  START → load_context → intent_route（条件分支）
    ├─ general_chat → result_synthesis → END
    └─ use_tool → tool_execute ─────────────────┘

阶段 6.2 将把 tool_execute 升级为 while-true 循环（多轮工具调用）。
阶段 6.1 先跑通最简单的：单工具调用 → 合成回复。
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from agent.state import MainState, StateField
from agent.nodes import load_context, result_synthesis
from agent.router import route_intent, get_route_key
from agent.tools.registry import get_tool, list_tools


# ── LLM 实例（模块级单例）─────────────────────────────────────

# TODO(human): 实现 get_llm() 函数
# 说明：
#   1. 从 rag.config.Config 读取 LLM 配置（MODEL_PROVIDER, MODEL_NAME, BASE_URL, API_KEY）
#   2. 用 langchain.chat_models.init_chat_model() 创建 ChatModel 实例
#      - 参数 model=Config.MODEL_NAME, model_provider=Config.MODEL_PROVIDER
#      - 参数 base_url=Config.BASE_URL, api_key=Config.API_KEY
#   3. 设置 temperature=0.7
#   4. 做成模块级懒加载单例：_llm 初始为 None，第一次调用时创建并缓存
#   5. 参考 sagt_agent/src/llm/llm_setting.py 的实现模式


# ── 工具执行节点 ──────────────────────────────────────────────

# TODO(human): 实现 tool_execute(state: MainState) -> dict 函数
# 说明：
#   1. 从 State 中取出 target_tool（路由节点已经确定了工具名）和 user_question
#   2. 调用 get_tool(target_tool) 获取工具函数
#   3. 构造 ToolContext(session_id=session_id)
#   4. 调用 tool_func(ctx, user_question) 执行工具
#   5. 返回 dict：
#      - StateField.TOOL_RESULTS → [result]（用列表包装，后续多工具时直接追加）
#      - StateField.NEED_CONTINUE → False（阶段 6.1 只调一次，6.2 升级为循环判断）
#   6. 异常处理：工具执行失败时也返回 ToolResult(success=False)，不中断流程


# ── 图构建 ────────────────────────────────────────────────────

# TODO(human): 实现 build_graph() -> CompiledStateGraph 函数
# 说明：
#   1. 创建 StateGraph(MainState)
#   2. 添加节点：
#      - "load_context" → load_context（来自 agent.nodes.load_context）
#      - "intent_route" → ... （调用 route_intent + 条件边，见思路 3）
#      - "tool_execute" → tool_execute
#      - "result_synthesis" → result_synthesis
#   3. 添加边：
#      - START → "load_context"
#      - "load_context" → "intent_route"
#      - intent_route 条件边：
#        - use_tool → "tool_execute"
#        - general_chat → "result_synthesis"
#      - "tool_execute" → "result_synthesis"
#      - "result_synthesis" → END
#   4. 编译时传入 checkpointer=MemorySaver()（内存状态持久化）
#   5. 返回编译后的 graph
#
#   关键 LangGraph API：
#     graph = StateGraph(MainState)
#     graph.add_node("name", function)
#     graph.add_edge(START, "name")
#     graph.add_conditional_edges("name", route_function, {目标: 下一节点})
#     return graph.compile(checkpointer=MemorySaver())
#
#   对 intent_route 节点的处理思路：
#     - 不能直接用 route_intent() 作为节点（它还需要 config 中的 model）
#     - 写一个薄封装函数 intent_node(state)，内部调用 route_intent + get_llm + list_tools
#     - 条件边用 intent_node → add_conditional_edges + get_route_key
#     - 目标映射：{"use_tool": "tool_execute", "general_chat": "result_synthesis"}


# ── 运行入口（调试用）─────────────────────────────────────────

# TODO(human): 实现 run_graph(session_id: str, question: str) -> str 函数
# 说明：
#   1. 调用 build_graph() 获取编译后的 graph
#   2. 构造初始 State：session_id + user_question
#   3. 调用 graph.invoke(initial_state, config={"configurable": {"thread_id": session_id}})
#      - thread_id 用 session_id，实现多会话隔离
#   4. 从返回的 State 中取出 final_response 并返回
#   5. 这个函数主要用来在终端快速测试 Graph 是否能跑通
