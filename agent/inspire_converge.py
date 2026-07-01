"""
灵感引擎 — Phase 2 收敛合成

职责：
  inspire_evidence + 用户问题 → "I think... and..." 风格合成 → 证据门检查 → final_response

数据流位置：
  inspire_diverge → 【本节点】 → memory_update
  （跳过通用 result_synthesis，因为灵感回复需要专属 prompt）
"""

from __future__ import annotations

import logging
import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from agent.bound import get_bound_summary
from agent.state import MainState, StateField
from agent.tools.base import ToolResult
from agent.trace import trace_inspire_converge
from db import save_message

logger = logging.getLogger(__name__)

# —— Prompt 模板 ───────────────────────────────────────────────

# 收敛合成：「I think... and...」风格，叠加延伸而非否定说教
CONVERGE_SYSTEM_PROMPT = """你是一位**科学作家**（Science Writer）——类似 Steven Johnson 或万维钢，
专长是从跨学科视角发现事物之间的隐藏联系，用平实但有深度的中文把复杂概念讲清楚。
你的写作风格是「串联洞察」而非「罗列事实」，读者读完应该觉得"原来如此，我之前没想到这个角度"。

## 用户画像
{memory_context}

## 你的风格（必须遵守）
1. 用「我觉得...，而且...」的方式**叠加延伸**用户的思路，不要否定或说教
2. 至少给出 **1 个跨领域连接**（把用户话题和另一个领域的理论/实践联系起来）
3. 至少给出 **1 个「用户可能不知道但值得知道」** 的观点或案例
4. 引用检索证据时自然融入，不要列 bullet 清单
5. 如果证据不足，坦诚说「这部分我找到的资料有限」，但仍尝试从已有知识延伸
6. 用中文，300~600 字，有思考深度但不冗长

{bound_summary}
"""

# 证据门：LLM 自检输出质量（Phase 1 不通过也仍输出，只记录 gate_passed=False）
EVIDENCE_GATE_PROMPT = """你是一位**学术期刊审稿人**（Peer Reviewer）——类似 Nature 或 Science 的同行评审专家。
你的职责不是判断观点对不对，而是检查论证是否扎实：有没有引用支撑？逻辑是否自洽？
你审稿时只看结构质量，不管你是否同意作者的观点。

## 最低标准
1. 有至少 1 个跨领域连接（不同学科/行业的关联）
2. 有至少 1 个引用检索证据或明确标注「基于已有知识延伸」
3. 风格是叠加延伸（I think... and...），不是否定说教

## 待审稿件
{draft}

## 可用检索证据摘要
{evidence_summary}

## 输出规则
严格输出 JSON，不要其他文字：
{{"passed": true/false, "reason": "一句话说明"}}

passed=true 表示 3 条标准都满足；任一不满足则 passed=false。
"""


# ── 1. 格式化 evidence ────────────────────────────────────────

def _format_evidence(evidence: list[ToolResult]) -> str:
    """把 ToolResult 列表格式化为 LLM 可读的证据文本。

    每条 evidence 标注来源工具和查询，方便 converge prompt 引用。

    Args:
        evidence: inspire_diverge 产出的搜索结果列表

    Returns:
        格式化后的多行文本
    """
    if not evidence:
        return "（未找到检索证据，请基于已有知识延伸）"

    lines: list[str] = []
    idx = 1

    for item in evidence:
        if not item.success:
            continue

        tool = (item.artifacts or {}).get("tool", "unknown")   # 工具名
        query = (item.artifacts or {}).get("query", "")        # 查询
        data_str = str(item.data) if item.data else ""         # 数据

        if not data_str.strip():    # 如果数据为空，则跳过
            continue

        snippet = data_str[:800] + ("..." if len(data_str) > 800 else "")  # 截断过长内容，控制 converge 阶段的 token
        lines.append(f"【证据{idx} | {tool} | 查询: {query}】\n{snippet}")
        idx += 1

    if not lines:
        return "（检索均未返回有效内容，请基于已有知识延伸）"

    return "\n\n".join(lines)
        

# —— 2. 收敛合成 ───────────────────────────────────────────────

def generate_inspire_response(
    question: str,
    memory_context: str,
    evidence_text: str,
    model: BaseChatModel,
) -> str:
    """基于证据生成 "I think... and..." 风格的灵感回复。

    Args:
        question: 用户原始问题
        memory_context: MEMORY.md 内容
        evidence_text: _format_evidence 产出的证据文本
        model: LangChain ChatModel 实例

    Returns:
        AI 回复文本
    """
    system_content = CONVERGE_SYSTEM_PROMPT.format(
        memory_context=memory_context.strip() or "（暂无）",
        bound_summary=get_bound_summary(),      # 系统约束摘要
    )

    user_content = (
        f"## 用户问题\n{question}\n\n"
        f"## 检索到的证据\n{evidence_text}\n\n"
        f"请用「我觉得...，而且...」的风格，给出有跨域连接的灵感回复。"
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ]

    response = model.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer.strip() if isinstance(answer, str) else str(answer).strip()

# —— 3. 证据门检查 ───────────────────────────────────────────────

def check_evidence_gate(
    draft: str,
    evidence_summary: str,
    model: BaseChatModel,
) -> tuple[bool, str]:
    """LLM 自检：回复是否满足跨域连接 + 证据引用 + 风格要求。

    Phase 1 策略：不通过也仍输出 draft，只记录 gate_passed=False。
    Phase 2 增强：不通过时触发补搜一轮。

    Args:
        draft: 收敛合成的回复草稿
        evidence_summary: 格式化后的证据摘要
        model: LangChain ChatModel 实例

    Returns:
        (gate_passed, reason) 元组
    """
    import json

    prompt = EVIDENCE_GATE_PROMPT.format(
        draft=draft[:1500],
        evidence_summary=evidence_summary[:2000],
    )

    try:
        response = model.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        raw = raw if isinstance(raw, str) else str(raw)
        raw = raw.strip()

        # 剥 ```json 包裹
        if raw.startswith("```"):
            lines = [l for l in raw.split("\n") if not l.strip().startswith("```")]
            raw = "\n".join(lines).strip()

        # 解析 JSON
        data = json.loads(raw)
        passed = bool(data.get("passed", False))
        reason = str(data.get("reason", ""))
        return passed, reason

    except Exception as e:
        logger.warning("证据门检查失败，默认放行: %s", e)
        return True, "gate_check_error_default_pass"


# —— 4. LangGraph 节点入口 ─────────────────────────────────────

def inspire_converge(state: MainState, model: BaseChatModel) -> dict:
    """灵感引擎收敛节点：合成回复 + 证据门 + 写 PG + 写入 final_response。
    
    读取：user_question, memory_context, session_id, trace_id, inspire_evidence
    写入：final_response

    注意：本节点替代 result_synthesis 的 PG 消息保存职责，
    因为 inspire 路径不经过 result_synthesis。

    Args:
        state: 当前 MainState
        model: LangChain ChatModel 实例

    Returns:
        dict: {FINAL_RESPONSE: str}
    """
    question = state.get(StateField.USER_QUESTION, "")
    memory_context = state.get(StateField.MEMORY_CONTEXT, "")
    session_id = state.get(StateField.SESSION_ID, "")
    trace_id = state.get(StateField.TRACE_ID, "")
    evidence: list[ToolResult] = state.get(StateField.INSPIRE_EVIDENCE, [])

    t0 = time.monotonic()

    # ① 格式化 evidence
    evidence_text = _format_evidence(evidence)
    
    # ② LLM 收敛合成
    draft = generate_inspire_response(
        question=question,
        memory_context=memory_context,
        evidence_text=evidence_text,
        model=model,
    )
    
    # ③ 证据门检查（Phase 1：不通过也输出 draft）
    gate_passed, gate_reason = check_evidence_gate(
        draft=draft,
        evidence_summary=evidence_text,
        model=model,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ④ H2 追踪打点
    if trace_id:
        trace_inspire_converge(
            trace_id,
            session_id,
            gate_passed=gate_passed,
            gate_reason=gate_reason,
            latency_ms=elapsed_ms,
        )

    # ⑤ 保存消息到 PG（result_synthesis 的职责，inspire 路径在此补做）
    try:
        save_message(session_id, "user", question)
        save_message(session_id, "assistant", draft)
    except Exception as e:
        logger.error("PG 消息保存失败: %s", e)

    # ⑥ 返回 State 更新
    return {
        StateField.FINAL_RESPONSE: draft,
    }