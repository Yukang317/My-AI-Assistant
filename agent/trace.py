"""
Agent 请求追踪 — 阶段 6.3 H2 Harness

双层设计（互补，不是二选一）：

  层 1 — LangSmith（LangChain/LangGraph 官方 SaaS）
    配好 .env 后，graph.invoke() + LangChain LLM 调用自动出 trace。
    适合：看完整 run 树、LLM 输入输出、节点耗时、网页 UI 回放。

  层 2 — 本地 JSONL（本项目自管）
    无论 LangSmith 开没开，都往 logs/agent_trace.jsonl 追加一行 JSON。
    适合：ECS 离线调试、grep 黄金路径、H3 测试对照、灵感引擎多阶段打点。

数据流（一次 /chat/graph 请求）：
  app.py run_graph()
    → trace_run_start()          # JSONL: run_start
    → graph.invoke()             # LangSmith 自动抓整图（若已启用）
      → intent_node
          → log_event(route)     # JSONL: route_decision（含 reason）
      → tool_execute（若有）
          → log_event(bound)     # JSONL: bound_check
          → log_event(tool)      # JSONL: tool_execute（含 latency_ms）
      → result_synthesis / memory_update
    → trace_run_end()            # JSONL: run_end（总 latency + 最终 intent/tool）

灵感引擎（6.5）扩展：event 字段追加 inspire_diverge / inspire_converge 即可，
不必重写本模块。
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator, Optional

# LangSmith 是 langchain 的传递依赖，无需单独 pip install
# 未配置 API Key 时，tracing_context 是空操作，不会报错
try:
    from langsmith import tracing_context       # 从 langsmith 包导入真正的 tracing_context
except ImportError:  # 极端环境兜底
    from contextlib import contextmanager as tracing_context  # 标准库的 contextmanager 装饰器改名叫 tracing_context（让它能当 context manager 用）

    @contextmanager  # 
    def tracing_context(**kwargs: Any) -> Iterator[None]:
        # 定义一个空的 tracing_context 函数——它什么都不做，只是占个位，让后面 with tracing_context(...): 那句不会报 NameError
        yield  

# ── 路径与开关 ────────────────────────────────────────────────

# personal_assistant/logs/agent_trace.jsonl
_PROJECT_ROOT = Path(__file__).resolve().parent.parent          # .resolve() 把 Path("agent/trace.py") 对象转换为绝对路径，parent.parent 得到项目根目录
TRACE_LOG_PATH = _PROJECT_ROOT / "logs" / "agent_trace.jsonl"

def is_langsmith_enabled() -> bool:
    """LangSmith 是否已配置为开启状态。

    官方文档要求同时设 LANGSMITH_TRACING=true 和 LANGSMITH_API_KEY。
    只设其一视为未启用，走纯本地 JSONL。
    """
    tracing = os.getenv("LANGSMITH_TRACING", "").strip().lower()
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    return tracing in ("true", "1", "yes") and bool(api_key)


def new_trace_id() -> str:
    """为单次 Graph 运行生成唯一 trace_id，串联 JSONL 多行事件。"""
    return uuid.uuid4().hex[:16]


# ── 事件模型 ──────────────────────────────────────────────────

@dataclass
class TraceEvent:
    """单条结构化追踪记录，序列化后写入 JSONL。
    
    Attributes:
        trace_id: 一次 graph.invoke 的唯一 ID，多行事件共享
        session_id: 用户会话 ID（= LangGraph thread_id）
        event: 事件类型，见模块 docstring
        timestamp: ISO8601 UTC 时间戳
        latency_ms: 本步骤耗时（毫秒），run_start 可为 None
        data: 业务字段（intent、tool、error 等），自由扩展
    """

    trace_id: str
    session_id: str
    event: str
    timestamp: str
    latency_ms: Optional[int] = None
    # 存业务数据
    data: dict[str, Any] = field(default_factory=dict)      #  每次创建新实例时，field 会调用 dict() 生成一个全新的空字典，各实例互不影响。


# ── 核心写入 ──────────────────────────────────────────────────

def _ensure_log_dir() -> None:
    """确保 logs/ 目录存在（首次写入时自动创建）。"""
    TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log_event(
    trace_id: str,
    session_id: str,
    event: str,
    *,                  # 强制 * 后续的参数必须以关键字参数形式传入，eg：“latency_ms=100, question="你好"” 而不是 “100, "你好"”
    latency_ms: Optional[int] = None,
    **data: Any,        # 收集所有多余的关键字参数
) -> None:
    """追加一行 JSON 到本地的 trace 日志

    所有打点都走这个函数，保证格式统一、方便 jq/grep

    Args:
        trace_id: 一次 graph.invoke 的唯一 ID，多行事件共享
        session_id: 用户会话 ID（= LangGraph thread_id）
        event: 事件名称（run_start / route_decision / bound_check / tool_execute / run_end）
        latency_ms: 本步骤耗时（毫秒），run_start 可为 None
        data: 任意业务字段（intent、tool、error 等），自由扩展
    """

    record = TraceEvent(
        trace_id=trace_id,
        session_id=session_id,
        event=event,
        timestamp=datetime.now(timezone.utc).isoformat(),
        latency_ms=latency_ms,
        data=data,
    )

    _ensure_log_dir()

    line = json.dumps(asdict(record), ensure_ascii=False)   # asdict 后变成普通字典
    # json.dumps({"msg": "你好"}, ensure_ascii=False)   # → '{"msg": "你好"}'
    # json.dumps({"msg": "你好"}, ensure_ascii=True)    # → '{"msg": "\\u4f60\\u597d"}'    确保非 ASCII 字符转义
    
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:      # "a" 表示末尾追加
        f.write(line + "\n")


# ── 语义化打点（main_graph 调用这些，不直接拼 event 名）──────

def trace_run_start(trace_id: str, session_id: str, question: str) -> None:
    """Graph 开始前：记录用户问题和 trace_id。"""
    # question 截断避免日志爆炸；完整内容 LangSmith UI 里看
    preview = question[:200] + ("..." if len(question) > 200 else "")
    log_event(trace_id, session_id, "run_start", question_preview=preview, langsmith_enabled=is_langsmith_enabled())
    # {
    # "event": "run_start",
    # "data": {
    #   "question_preview": "你好，我想问...",
    #   "langsmith_enabled": true
    # }
    # }



def trace_route_decision(
    trace_id: str,
    session_id: str,
    *,
    intent: str,
    target_tool: str,
    reason: str,
    latency_ms: int,
) -> None:
    """intent_node 完成后：记录路由决策（H4 搁置后仍可用于分析路由质量）。"""
    log_event(trace_id, session_id, "route_decision", intent=intent, target_tool=target_tool, reason=reason, latency_ms=latency_ms)

def trace_bound_check(
    trace_id: str,
    session_id: str,
    *,
    target_tool: str,
    allowed: bool,
    reason: str = "",
) -> None:
    """check_bound 结果：H1 安检是否放行。"""
    log_event(trace_id, session_id, "bound_check", target_tool=target_tool, allowed=allowed, reason=reason)

def trace_tool_execute(
    trace_id: str,
    session_id: str,
    *,
    target_tool: str,
    success: bool,
    latency_ms: int,
    error: str = "",
) -> None:
    """工具执行完毕：记录成败和耗时（灵感引擎多工具时会多次调用）。"""
    log_event(trace_id, session_id, "tool_execute", target_tool=target_tool, success=success, latency_ms=latency_ms, error=error)

def trace_run_end(
    trace_id: str,
    session_id: str,
    *,
    latency_ms: int,
    intent: str,
    target_tool: str,
    final_response_preview: str = "",
    error: str = "",
) -> None:
    """Graph 整轮结束：汇总总耗时和最终状态。"""
    preview = final_response_preview[:200] + (
        "..." if len(final_response_preview) > 200 else ""
    )
    log_event(
        trace_id,
        session_id,
        "run_end",
        latency_ms=latency_ms,
        intent=intent,   
        target_tool=target_tool,
        final_response_preview=preview,
        error=error,
    )


# ── Graph 运行上下文（run_graph / run_graph_debug 用）────────

@contextmanager
def agent_trace_context(
    session_id: str,
    question: str,
) -> Generator[dict[str, Any], None, None]:
    """一次 Graph 调用的追踪上下文。

    用法（在 main_graph.run_graph 里）：
        with agent_trace_context(session_id, question) as ctx:
            ctx["trace_id"]  # 传给各节点打点
            final_state = graph.invoke(..., config=langsmith_config(ctx))

    Yields:
        dict: 至少含 trace_id、session_id、question、t0（monotonic 起点）
    """
    trace_id = new_trace_id()           # 1. 生成本次 trace ID
    t0 = time.monotonic()               # 2. 记下开始时间，高精度计时

    # 先签到再执行，防止graph中途崩溃而没有任何信息。=======自定义的trace追踪=======
    trace_run_start(trace_id, session_id, question) # 3. 写 run_start 日志

    #   - metadata={"session_id": "xxx"} 告诉 LangSmith："这些 trace 是一家的"
    #   - 然后在 LangSmith UI 里就能按 session_id 筛选，看到一个会话的完整对话链
    # thread_id 键名是 LangSmith 文档推荐的 conversation 分组字段之一

    # 资源获取即初始化（RAII）：with 能保证进入前后自动做准备和清理工作，中间不论状态均执行清理。
    with tracing_context(               # 4. 进入 LangSmith 追踪上下文。===== LangSmith官方的trace追踪 =====
        metadata={
            "session_id": session_id,
            "trace_id": trace_id,
        },
        tags=["personal_assistant", "agent_graph"],
    ):
        yield {                       # 5. 暂停，把控制权交出去。返回追踪上下文，包含 trace_id、session_id、question、t0  
            "trace_id": trace_id,
            "session_id": session_id,
            "question": question,
            "t0": t0,
        }

    # ⑥ with 块结束后回到这里
    # ⑦ tracing_context 退出，LangSmith 追踪关闭
    # ⑧ 注意：这里不调 trace_run_end！调用方自己调
    # run_end 由调用方（也就是graph）在 invoke 完成后显式调 trace_run_end（需要 final_state）

def read_recent_traces(limit: int = 20) -> list[dict[str, Any]]:
    """读取最近 N 条 JSONL 记录（调试 / 测试辅助）。

    Args:
        limit: 最多返回条数

    Returns:
        解析后的 dict 列表，文件不存在时返回空列表
    """
    if not TRACE_LOG_PATH.exists():
        return []
    lines = TRACE_LOG_PATH.read_text(encoding="utf-8").strip().splitlines()
    records = [json.loads(line) for line in lines[-limit:]]
    return records



# ── 灵感引擎扩展 ──────────────────────────────────────────────────

def trace_inspire_diverge(
    trace_id: str,
    session_id: str,
    *,
    queries: list[str],
    evidence_count: int,
    latency_ms: int,
) -> None:
    """灵感引擎发散：记录多角度查询和检索证据。"""
    log_event(trace_id, session_id, "inspire_diverge", queries=queries, evidence_count=evidence_count, latency_ms=latency_ms)
    # {
    # "event": "inspire_diverge",
    # "data": {
    #   "queries": ["认知科学启发", "跨界类比"],
    #   "evidence_count": 2,
    #   "latency_ms": 100
    # }
    # }

def trace_inspire_converge(
    trace_id: str,
    session_id: str,
    *,
    gate_passed: bool,
    gate_reason: str,
    latency_ms: int,
) -> None:
    """灵感引擎收敛：记录证据门结果和耗时。"""
    log_event(trace_id, session_id, "inspire_converge", gate_passed=gate_passed, gate_reason=gate_reason, latency_ms=latency_ms)
    # {
    # "event": "inspire_converge",
    # "data": {
    #   "gate_passed": true,
    #   "gate_reason": "证据门通过",
    #   "latency_ms": 100
    # }
    # }