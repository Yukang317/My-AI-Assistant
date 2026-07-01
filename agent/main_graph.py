"""
主图构建 — LangGraph StateGraph 组装

架构：
  START → load_context → intent_route（条件分支）
    ├─ general_chat → result_synthesis → memory_update → END
    └─ use_tool → tool_execute → result_synthesis → memory_update → END

阶段 6.3 H2：在 H1 基础上接入 agent/trace.py 双层追踪。
阶段 6.2 将把 tool_execute 升级为 while-true 循环（多轮工具调用）。
阶段 6.1 先跑通最简单的：单工具调用 → 合成回复。
"""

import time
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
# ★ H1 新增 import：BOUND 检查 + 查工具安全分类
from agent.bound import check_bound
from agent.tools.registry import get_tool, get_tool_bound, list_tools
from agent.tools.base import ToolContext, ToolResult

# ★ H2 新增：结构化追踪（本地 JSONL + LangSmith tracing_context）
from agent.trace import (
    agent_trace_context,        # 自定义的trace追踪
    trace_bound_check,          # 安检
    trace_route_decision,       # 路由
    trace_run_end,              # 运行结束
    trace_tool_execute,         # 工具执行
)

# ★ H3 新增：灵感引擎扩展（发散-收敛管线）
from agent.inspire_diverge import inspire_diverge
from agent.inspire_converge import inspire_converge


# ── 循环预算默认值 ────────────────────────────────────────────

DEFAULT_MAX_TURNS = 3           # 最大循环次数
DEFAULT_TIME_BUDGET_MS = 30_000 # 总超时预算30秒
# 软 token 预算：每轮估算 len(question)+len(data)，累加超阈值则停
DEFAULT_TOKEN_BUDGET_ESTIMATE = 8_000 # 软 token 预算8000


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


# 内部判断函数，基于工具结果决定是否继续调用
def _should_continue_by_observation(
  question: str,
  latest_tool_result: ToolResult,
  turn_count: int,
  max_turns: int,
  elapsed_ms: int,
  time_budget_ms: int,
  model: BaseChatModel,
) -> tuple[bool, str]:
  """规则优先 + LLM 兜底：判断是否需要继续调用工具
  
  Args:
    question: 用户问题
    latest_tool_result: 最新工具结果
    turn_count: 当前循环次数
    max_turns: 最大循环次数   
    elapsed_ms: 当前累计耗时
    time_budget_ms: 总超时预算
    model: LangChain ChatModel 实例
  
  Returns:
    tuple[bool, str]: 是否继续调用，继续调用原因
  """

  # 规则 1：达到最大次数
  if turn_count >= max_turns:
    return False, "reach_max_turns"

  # 规则 2：工具执行失败
  if not latest_tool_result.success:
    return False, "tool_failed"

  # 规则 3：超时
  if elapsed_ms >= time_budget_ms:
    return False, "timeout"

  # 规则 4：结果已充分（>80 字符）
  data_str = str(latest_tool_result.data) if latest_tool_result.data else ""
  if len(data_str) > 80:
    return False, "enough_information"

  # 规则 5：LLM 兜底
  prompt = (
    f"用户问题：{question}\n"
    f"工具返回内容：{data_str[:500]}\n\n"
    f"以上信息是否足够回答用户问题？\n"
    f"如果足够，只回复 STOP。如果还需要更多信息，只回复 CONTINUE。\n"
    f"不要解释，不要任何其他文字。"
  )
  try:
    response = model.invoke(prompt)
    if hasattr(response, "content"):
      answer = response.content
    else:
      answer = str(response)
    answer = answer.strip().upper()
  except Exception:
    return False, "llm_error"

  if "CONTINUE" in answer:
    return True, "llm_continue"
  return False, "llm_stop"


# ── Harness H1：BOUND 分类 → check_bound 用的动作描述 ──────────
# NEVER_DO 黑名单里有「发送HTTP请求」，NETWORK 工具必须用「网络搜索」绕过
BOUND_ACTION_MAP: dict[str, str] = {
    "READ_ONLY": "读取数据",
    "NETWORK": "网络搜索",
    "WRITE": "写入数据",
}



# ── 工具执行节点 ──────────────────────────────────────────────

def tool_execute(state: MainState) -> dict:
    """执行路由选中的工具，返回 ToolResult 列表。

    intent_route 已经确定了 target_tool 并写入 State，
    阶段 6.3 H1：先过 BOUND 确定性检查，再真正调用工具。
    
    Args:
        state: 当前 MainState，读取 target_tool + user_question + session_id
    
    Returns:
        dict: {TOOL_RESULTS: [ToolResult], NEED_CONTINUE: False}
    """
    target_tool = state.get(StateField.TARGET_TOOL, "")
    question = state.get(StateField.USER_QUESTION, "")
    session_id = state.get(StateField.SESSION_ID, "")

    # H2：从 State 取 trace_id（invoke 前由 _invoke_with_trace 写入）
    trace_id = state.get(StateField.TRACE_ID, "")

    # H2：从 State 取 loop 控制变量
    # 循环控制变量，从 State 取，给默认值
    max_turns = state.get(StateField.MAX_TURNS, DEFAULT_MAX_TURNS)  # 最大循环次数
    time_budget_ms = state.get(StateField.TIME_BUDGET_MS, DEFAULT_TIME_BUDGET_MS) # 总超时预算30秒
    token_budget_limit = state.get(
      StateField.TOKEN_BUDGET_ESTIMATE, DEFAULT_TOKEN_BUDGET_ESTIMATE
    ) # 软 token 预算8000

    # 初始化：结果列表 + 计时起点 + 软 token 累计
    tool_results_list: list[ToolResult] = []
    turn_count = 0
    token_budget_used_estimate = 0
    node_t0 = time.monotonic()
    loop_stop_reason = "unknown"
    model = get_llm()


    # H2：循环执行工具调用，直到达到最大循环次数或超时
    while turn_count < max_turns:

      # ── 第 1 步：Harness 安检（确定性代码，不信任 LLM）──
      try:
        # 从注册表查这个工具的安全分类（READ_ONLY / NETWORK / WRITE）
        bound_category = get_tool_bound(target_tool)
        # 把分类映射成 action 描述；NETWORK →「网络搜索」，不会命中 NEVER_DO
        action_desc = BOUND_ACTION_MAP.get(bound_category, "未知操作")
        action = f"调用工具:{target_tool} {action_desc}"
        # target 用用户问题；若问题里含 .env 等危险路径模式会被 DANGER_ZONES 拦截
        allowed, reason = check_bound(action, question)

        # H2：记录安检结果（无论通过与否）
        if trace_id:
          trace_bound_check(
            trace_id,
            session_id,
            target_tool=target_tool,
            allowed=allowed,
            reason=reason,
          )

        if not allowed:
          loop_stop_reason = "bound_rejected"
          tool_results_list.append(ToolResult(success=False, data="", error=reason))
          break   # 不 return，统一走底部的返回值逻辑

      except Exception as e:
        if trace_id:
          trace_bound_check(
            trace_id,
            session_id,
            target_tool=target_tool,
            allowed=False,
            reason=str(e),
          )

        loop_stop_reason = "tool_setup_error"
        tool_results_list.append(ToolResult(success=False, data="", error=str(e)))
        break   # 不 return，统一走底部的返回值逻辑

      # ── 第 2 步：安检通过，真正执行工具（和原来一样）──
      t0 = time.monotonic()       # 执行工具 + 计时
      try:
        # 返回值是一个**从字典里取出来的**函数，接收 ToolContext + 输入字符串，返回 ToolResult
        tool_func = get_tool(target_tool)           # 从注册表拿函数
        ctx = ToolContext(session_id=session_id)    # 构造调用上下文
        # 执行**取出来的那个函数**
        result = tool_func(ctx, question)           # 执行！result 是 ToolResult
      except Exception as e:
        result = ToolResult(success=False, data="", error=str(e))
      
      step_ms = int((time.monotonic() - t0) * 1000)   # 执行工具的耗时，单位毫秒


      # —— 第 3 步：累积结果 ——
      tool_results_list.append(result)
      

      # —— 第 4 步：trace ——
      if trace_id:
        trace_tool_execute(
          trace_id,
          session_id,
          target_tool=target_tool,
          success=result.success,
          latency_ms=step_ms,
          error=result.error or "",
        )

      # 第 5 步：turn_count += 1（放在工具执行之后）
      turn_count += 1

      # 第 5.5 步：token 软预算（每轮估算 question + data 字符数）
      data_str = str(result.data) if result.data else ""
      token_budget_used_estimate += len(question) + len(data_str)
      if token_budget_used_estimate >= token_budget_limit:
        loop_stop_reason = "budget_exceeded"
        break

      # —— 第 6 步：判断是否继续 ——
      total_elapsed = int((time.monotonic() - node_t0) * 1000)
      should_continue, reason = _should_continue_by_observation(
        question=question,
        latest_tool_result=result,
        turn_count=turn_count,
        max_turns=max_turns,
        elapsed_ms=total_elapsed,
        time_budget_ms=time_budget_ms,
        model=model,
      )

      if not should_continue:       # 局部变量，while 循环要不要继续
        loop_stop_reason = reason
        break   # 不 return，统一走底部的返回值逻辑，包括 break 和 return

    if loop_stop_reason == "unknown":
      loop_stop_reason = "reach_max_turns"

    total_elapsed_ms = int((time.monotonic() - node_t0) * 1000)

    return {
      StateField.TOOL_RESULTS: tool_results_list,
      StateField.NEED_CONTINUE: False,                  # Graph 路由不回头 -> 不继续调用工具
      StateField.TURN_COUNT: turn_count,
      StateField.LOOP_STOP_REASON: loop_stop_reason,
      StateField.TOOL_BUDGET_USED: turn_count,
      StateField.TOKEN_BUDGET_USED_ESTIMATE: token_budget_used_estimate,
      StateField.ELAPSED_MS: total_elapsed_ms,
    }

# ── 图构建 ────────────────────────────────────────────────────

def build_graph() -> CompiledStateGraph:
  """构建并编译 Agent 主图。

  节点注册 → 边连接 → 条件路由 → 编译。
  返回编译后的图，可直接 invoke() 执行。

  当前图结构：
    START → load_context → intent_route（条件分支）
      ├─ general_chat → result_synthesis → memory_update → END
      ├─ use_tool → tool_execute → result_synthesis → memory_update → END
      └─ inspire → inspire_diverge → inspire_converge → memory_update → END
  """
  # 1. 创建图
  graph = StateGraph(MainState)

  # 2. 包装节点 — 桥接 LangGraph 节点签名和业务函数签名
  def intent_node(state: MainState) -> dict:
    """适配层：把 LangGraph 节点签名转成 route_intent 的调用格式。"""
    question = state.get(StateField.USER_QUESTION, "")
    model = get_llm()
    tools = list_tools()        # 所有可用的工具的名字 + 描述
    trace_id = state.get(StateField.TRACE_ID, "")
    session_id = state.get(StateField.SESSION_ID, "")

    t0 = time.monotonic()       # 起点时间戳，单位 s，精确到微秒
    result: RouteResult = route_intent(question, model, tools)
    elapsed_ms = int((time.monotonic() - t0) * 1000)    # 经过的秒数转换为毫秒并去掉小数

    if trace_id:
      trace_route_decision(
        trace_id,
        session_id,
        intent=result.intent,
        target_tool=result.target_tool or "",
        reason=result.reason,
        latency_ms=elapsed_ms,
      )
    
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

  def inspire_diverge_node(state: MainState) -> dict:
    """适配层：把 LangGraph 节点转成 inspire_diverge 的调用格式。"""
    return inspire_diverge(state, get_llm())

  def inspire_converge_node(state: MainState) -> dict:
    """适配层：把 LangGraph 节点转成 inspire_converge 的调用格式。"""
    return inspire_converge(state, get_llm())

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
  graph.add_node("inspire_diverge", inspire_diverge_node)
  graph.add_node("inspire_converge", inspire_converge_node)

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
      "inspire": "inspire_diverge",          # 灵感发散：P1 暂走工具调用（P2 升级为发散-收敛管线）
      "general_chat": "result_synthesis", # 普通闲聊：直接 LLM 回复
    },
  )
  graph.add_edge("tool_execute", "result_synthesis") # 固定边：工具→合成
  graph.add_edge("inspire_diverge", "inspire_converge") # 固定边：发散→收敛
  graph.add_edge("inspire_converge", "memory_update") # 固定边：收敛→记忆更新
  graph.add_edge("result_synthesis", "memory_update") # 固定边：合成→记忆更新
  graph.add_edge("memory_update", END) # 固定边：记忆更新→终点

  # 5. 编译（带内存检查点）-有了 checkpointer 后：每次执行完一个节点，LangGraph 自动把当前 State "拍照存档"。下次用同一个 thread_id（也就是 session_id）调用时，它从存档点继续，而不是从零开始。
  return graph.compile(checkpointer=MemorySaver())


# ── H2：带追踪的 invoke 封装 ─────────────────────────────────
def _invoke_with_trace(session_id: str, question: str) -> dict:
  """执行 graph.invoke 并写入  run_start / run_end 追踪事件。"""
  graph = build_graph()

  # trace_id 写入 initial_state，各节点从 State 读取后打点
  with agent_trace_context(session_id, question) as ctx:
    trace_id = ctx["trace_id"]
    t0 = ctx["t0"]

    initial_state: MainState = {
      StateField.SESSION_ID: session_id,
      StateField.USER_QUESTION: question,
      StateField.TRACE_ID: trace_id,      # ★ 需要 state.py 新增此字段
    }

    try:
      final_state = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
      )
      error = ""
    except Exception as e:
      # 异常也记 run_end，方便 H3 对照
      elapsed_ms = int((time.monotonic() - t0) * 1000)
      trace_run_end(
        trace_id,
        session_id,
        latency_ms=elapsed_ms,
        intent="",
        target_tool="",
        error=str(e),
      )
      raise

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    trace_run_end(
      trace_id,
      session_id,
      latency_ms=elapsed_ms,
      intent=final_state.get(StateField.INTENT, ""),
      target_tool=final_state.get(StateField.TARGET_TOOL, ""),
      final_response_preview=final_state.get(StateField.FINAL_RESPONSE, ""),
    )

    return final_state


# ── 运行入口（调试用）─────────────────────────────────────────

def run_graph_debug(session_id: str, question: str) -> dict:
  """调试入口：返回完整 State + H2 追踪已写入 JSONL。"""
  return _invoke_with_trace(session_id, question)


def run_graph(session_id: str, question: str) -> str:
  """生产入口：/chat/graph 调用，返回 AI 回复文本。"""
  final_state = _invoke_with_trace(session_id, question)
  return final_state.get(StateField.FINAL_RESPONSE, "")


