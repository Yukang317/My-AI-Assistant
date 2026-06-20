"""
主图构建 — LangGraph StateGraph 组装

架构：
  START → load_context → intent_route（条件分支）
    ├─ general_chat → result_synthesis → memory_update → END
    └─ use_tool → tool_execute → result_synthesis → memory_update → END

阶段 6.2 将把 tool_execute 升级为 while-true 循环（多轮工具调用）。
阶段 6.1 先跑通最简单的：单工具调用 → 合成回复。
"""

from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from rag.config import Config

from agent.state import MainState, StateField
from agent.nodes import load_context, result_synthesis
from agent.nodes.memory_update import memory_update
from agent.router import route_intent, get_route_key, RouteResult
from agent.tools.registry import get_tool, list_tools
from agent.tools.base import ToolContext, ToolResult

# ── LLM 实例（模块级单例）─────────────────────────────────────

_llm = None   # 模块级缓存

# get_llm() 在 main_graph.py 里是单例——整个 Graph 共用同一个 LLM 实例，配置来自 .env（DeepSeek API Key 等）。
def get_llm() -> BaseChatModel:
  """懒加载 LLM 单例，和 sagt_agent 的 llm_setting.py 同模式。
    
  和 RagService 共享 Config 中的 LLM 配置（DEEPSEEK_API_KEY 等），
  不额外引入新的环境变量。
  """
  global _llm
  if _llm is not None:
    return _llm

  _llm = init_chat_model(
    model=Config.LLM_MODEL,
    model_provider="openai",
    base_url=Config.LLM_BASE_URL,
    api_key=Config.LLM_API_KEY,
    temperature=Config.LLM_TEMPERATURE,   # 0.1 RAG场景偏保守
  )

  return _llm





# ── 工具执行节点 ──────────────────────────────────────────────

def tool_execute(state: MainState) -> dict:
    """执行路由选中的工具，返回 ToolResult 列表。

    intent_route 已经确定了 target_tool 并写入 State，
    本节点只负责取工具 → 构造 ToolContext → 调用 → 包装返回值。
    
    Args:
        state: 当前 MainState，读取 target_tool + user_question + session_id
    
    Returns:
        dict: {TOOL_RESULTS: [ToolResult], NEED_CONTINUE: False}
    """
    target_tool = state.get(StateField.TARGET_TOOL, "")
    question = state.get(StateField.USER_QUESTION, "")
    session_id = state.get(StateField.SESSION_ID, "")

    try:
      tool_func = get_tool(target_tool)           # 从注册表拿函数
      ctx = ToolContext(session_id=session_id)    # 构造调用上下文
      result = tool_func(ctx, question)           # 执行！result 是 ToolResult
    except Exception as e:
      result = ToolResult(success=False, error=str(e))  # 兜底，Graph不崩
    
    return {
      StateField.TOOL_RESULTS: [result],                        # 列表包装，result_synthesis 会遍历
      StateField.NEED_CONTINUE: False,                          # 阶段 6.1 固定 False，6.2 改循环判断
    }



# ── 图构建 ────────────────────────────────────────────────────

def build_graph() -> CompiledStateGraph:
  """构建并编译 Agent 主图。

  节点注册 → 边连接 → 条件路由 → 编译。
  返回编译后的图，可直接 invoke() 执行。

  当前图结构：
    START → load_context → intent_route（条件分支）
      ├─ general_chat → result_synthesis → END
      ├─ use_tool → tool_execute → result_synthesis → END
      └─ inspire → tool_execute → result_synthesis → END
        （P1: inspire 临时走单工具调用；P2/P3 升级为 Phase 1 发散+Phase 2 收敛管线）
  """
  # 1. 创建图
  graph = StateGraph(MainState)

  # 2. 包装节点 — 桥接 LangGraph 节点签名和业务函数签名
  def intent_node(state: MainState) -> dict:
    """适配层：把 LangGraph 节点签名转成 route_intent 的调用格式。"""
    question = state.get(StateField.USER_QUESTION, "")
    model = get_llm()
    tools = list_tools()        # 所有可用的工具的名字 + 描述
    result: RouteResult = route_intent(question, model, tools)
    return {
      StateField.INTENT: result.intent,
      StateField.TARGET_TOOL: result.target_tool or "",
    }

  def result_synthesis_node(state: MainState) -> dict:
    """适配层：给 result_synthesis 注入 model（依赖注入 → LangGraph 节点）。"""
    return result_synthesis(state, get_llm())

  def memory_update_node(state: MainState) -> dict:
    """适配层：把 LangGraph 节点签名转成 memory_update 的调用格式。"""
    return memory_update(state, get_llm())

  '''
  LangGraph 注册节点时，只认这种函数：
  def 某个节点(state: MainState) -> dict:
      ...
  '''

  # 3. 注册节点
  graph.add_node("load_context", load_context)
  graph.add_node("intent_route", intent_node)
  graph.add_node("tool_execute", tool_execute)
  graph.add_node("result_synthesis", result_synthesis_node)
  graph.add_node("memory_update", memory_update_node)

  # 4. 连接边
  graph.add_edge(START, "load_context")             # 固定边：起点→加载上下文
  graph.add_edge("load_context", "intent_route")    # 固定边：上下文→路由

  # 条件边：根据 intent 路由到不同分支
  # get_route_key 现在同时检查 intent 和 target_tool
  graph.add_conditional_edges(
    "intent_route",
    lambda state: get_route_key(
      intent=state.get(StateField.INTENT, "general_chat"),
      target_tool=state.get(StateField.TARGET_TOOL),
    ),
    {
      "use_tool": "tool_execute",         # 事实查询：单工具调用 → 基于结果回答
      "inspire": "tool_execute",          # 灵感发散：P1 暂走工具调用（P2 升级为发散-收敛管线）
      "general_chat": "result_synthesis", # 普通闲聊：直接 LLM 回复
    },
  )
  graph.add_edge("tool_execute", "result_synthesis") # 固定边：工具→合成
  graph.add_edge("result_synthesis", "memory_update") # 固定边：合成→记忆更新
  graph.add_edge("memory_update", END) # 固定边：记忆更新→终点

  # 5. 编译（带内存检查点）-有了 checkpointer 后：每次执行完一个节点，LangGraph 自动把当前 State "拍照存档"。下次用同一个 thread_id（也就是 session_id）调用时，它从存档点继续，而不是从零开始。
  return graph.compile(checkpointer=MemorySaver())


# ── 运行入口（调试用）─────────────────────────────────────────

def run_graph_debug(session_id: str, question: str) -> dict:
  """调试入口：返回完整 State，方便查看意图、路由、工具结果等中间态。

  与 run_graph() 的区别：返回整个 final_state dict 而不是只提取 final_response。
  测试脚本用这个来打印各阶段详情。
  """
  graph = build_graph()
  initial_state: MainState = {
    StateField.SESSION_ID: session_id,
    StateField.USER_QUESTION: question,
  }
  return graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": session_id}},
  )


def run_graph(session_id: str, question: str) -> str:
  """构建图 + 执行一次对话，返回 AI 回复。

  这是调试入口——在终端快速验证整个 Agent 链路是否跑通。
  生产环境中 Sidebar 的 /chat 端点也会调用类似的逻辑。

  Args:
      session_id: 会话ID，用于多轮对话的检查点隔离
      question: 用户当前问题
  
  Returns:
      AI 回复文本
  """
  graph = build_graph()                                # 获取编译后的图

  initial_state: MainState = {
    StateField.SESSION_ID: session_id,
    StateField.USER_QUESTION: question,
  }

  final_state = graph.invoke(
    initial_state,
    config = {"configurable": {"thread_id": session_id}},
  )

  return final_state.get(StateField.FINAL_RESPONSE, "")