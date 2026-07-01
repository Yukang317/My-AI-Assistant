"""
阶段 6.5 Inspire 专项测试 — 用 mock 覆盖发散/收敛核心逻辑

运行：
  cd personal_assistant && PYTHONPATH=. uv run python test/agent_inspire_test260626.py

设计原则：
  - 不调真实 LLM、不连数据库、不走网络 → 纯 mock，秒级跑完
  - 直接测试 inspire_diverge / inspire_converge 函数及关键子函数
  - 用 unittest.mock.patch 替换 get_tool / check_bound / model.invoke / save_message

覆盖场景：
  1. _parse_queries_json 正常 JSON 解析
  2. _parse_queries_json 非法 JSON → 兜底原问题
  3. execute_parallel_searches → 2 query × 2 tool = 4 条 evidence + artifacts
  4. BOUND 拒绝 → 单条 success=False，其余正常
  5. _format_evidence 过滤失败项，保留成功项
  6. inspire_diverge 节点 → 写入 inspire_queries + inspire_evidence
  7. inspire_converge + gate 通过 → final_response 含草稿
  8. inspire_converge + gate 不通过 → 仍输出 draft（Phase 1 策略）
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from agent.inspire_converge import (
    _format_evidence,
    inspire_converge,
)
from agent.inspire_diverge import (
    _parse_queries_json,
    execute_parallel_searches,
    inspire_diverge,
)
from agent.state import MainState, StateField
from agent.tools.base import ToolContext, ToolResult


# ── 测试结果数据结构 ──────────────────────────────────────────────

@dataclass
class InspireTestCase:
    """单条 inspire 测试用例。"""
    name: str
    description: str


@dataclass
class InspireTestResult:
    """单条测试的执行结果。"""
    case: InspireTestCase
    passed: bool
    message: str


# ── Mock 辅助函数 ─────────────────────────────────────────────────

def build_mock_llm(*responses: str) -> MagicMock:
    """构造假 LLM，按顺序返回预设 content。"""
    if not responses:
        raise ValueError("至少传一个响应")

    call_count = [0]

    def side_effect(_prompt) -> MagicMock:
        idx = min(call_count[0], len(responses) - 1)
        content = responses[idx]
        call_count[0] += 1
        return MagicMock(content=content)

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = side_effect
    return mock_llm


def build_mock_tool_factory(prefix: str = "mock"):
    """返回 get_tool 可用的 side_effect：按工具名区分返回数据。"""

    def get_tool_fn(tool_name: str):
        def mock_tool(ctx: ToolContext, query: str) -> ToolResult:
            return ToolResult(
                success=True,
                data=f"{prefix}-{tool_name}-{query[:12]}",
            )
        return mock_tool

    return get_tool_fn


def build_base_state(**overrides) -> MainState:
    """构造 inspire 节点所需的最小 MainState。"""
    base: MainState = {
        StateField.SESSION_ID: "test-inspire-session",
        StateField.USER_QUESTION: "如何用认知科学改进我的 RAG 系统？",
        StateField.MEMORY_CONTEXT: "用户是 AI 开发者",
        StateField.TRACE_ID: "",
        StateField.HISTORY_MESSAGES: [],
        StateField.INTENT: "inspire",
    }
    base.update(overrides)
    return base


# ── 测试用例实现 ──────────────────────────────────────────────────

def run_inspire_test(case: InspireTestCase) -> InspireTestResult:
    """执行单条 inspire 测试。"""
    name = case.name

    if name == "parse_queries_valid":
        raw = '{"queries": ["认知科学 记忆提取", "跨界类比 RAG", "反直觉视角"]}'
        result = _parse_queries_json(raw, fallback_question="兜底", max_queries=3)
        if result == ["认知科学 记忆提取", "跨界类比 RAG", "反直觉视角"]:
            return InspireTestResult(case, True, f"解析 3 条 query ✓")
        return InspireTestResult(case, False, f"期望 3 条，实际 {result}")

    if name == "parse_queries_fallback":
        result = _parse_queries_json("not json at all", fallback_question="原问题", max_queries=3)
        if result == ["原问题"]:
            return InspireTestResult(case, True, "非法 JSON 兜底原问题 ✓")
        return InspireTestResult(case, False, f"期望 ['原问题']，实际 {result}")

    if name == "parallel_search_count":
        queries = ["查询A", "查询B"]
        with patch("agent.inspire_diverge.get_tool", side_effect=build_mock_tool_factory()), \
             patch("agent.inspire_diverge.get_tool_bound", return_value="READ_ONLY"), \
             patch("agent.inspire_diverge.check_bound", return_value=(True, "")):
            evidence = execute_parallel_searches(
                queries=queries,
                session_id="sess-parallel",
                tools=("rag_search", "exa_search"),
                max_workers=4,
            )

        if len(evidence) != 4:
            return InspireTestResult(case, False, f"期望 4 条 evidence，实际 {len(evidence)}")

        tools = {(e.artifacts or {}).get("tool") for e in evidence}
        if tools != {"rag_search", "exa_search"}:
            return InspireTestResult(case, False, f"工具来源不完整: {tools}")

        if not all(e.success for e in evidence):
            return InspireTestResult(case, False, "存在失败的搜索结果")

        return InspireTestResult(case, True, "2×2 并行搜索 + artifacts ✓")

    if name == "bound_rejected":
        call_log: list[str] = []

        def mock_check_bound(action: str, query: str) -> tuple[bool, str]:
            call_log.append(query)
            if query == "拒绝这条":
                return False, "模拟拒绝"
            return True, ""

        with patch("agent.inspire_diverge.get_tool", side_effect=build_mock_tool_factory()), \
             patch("agent.inspire_diverge.get_tool_bound", return_value="NETWORK"), \
             patch("agent.inspire_diverge.check_bound", side_effect=mock_check_bound):
            evidence = execute_parallel_searches(
                queries=["正常查询", "拒绝这条"],
                session_id="sess-bound",
                tools=("exa_search",),
                max_workers=2,
            )

        rejected = [e for e in evidence if not e.success and "BOUND" in (e.error or "")]
        success = [e for e in evidence if e.success]
        if len(rejected) != 1 or len(success) != 1:
            return InspireTestResult(
                case, False,
                f"期望 1 拒绝 + 1 成功，实际 rejected={len(rejected)} success={len(success)}",
            )
        return InspireTestResult(case, True, "BOUND 拒绝单条，其余正常 ✓")

    if name == "format_evidence_filters":
        evidence = [
            ToolResult(success=False, data="", error="fail"),
            ToolResult(success=True, data="", artifacts={"tool": "rag_search", "query": "空数据"}),
            ToolResult(
                success=True,
                data="有效证据内容",
                artifacts={"tool": "exa_search", "query": "跨界类比"},
            ),
        ]
        text = _format_evidence(evidence)
        if "有效证据内容" not in text:
            return InspireTestResult(case, False, "成功证据未出现在格式化文本中")
        if "fail" in text.lower() or "空数据" in text:
            return InspireTestResult(case, False, "失败/空数据项未被过滤")
        if "exa_search" not in text:
            return InspireTestResult(case, False, "来源工具标注缺失")
        return InspireTestResult(case, True, "过滤失败项，保留成功证据 ✓")

    if name == "diverge_node_state":
        mock_llm = build_mock_llm(
            '{"queries": ["角度1", "角度2"]}',
        )
        state = build_base_state(trace_id="")  # 无 trace，跳过打点

        with patch("agent.inspire_diverge.get_tool", side_effect=build_mock_tool_factory("div")), \
             patch("agent.inspire_diverge.get_tool_bound", return_value="READ_ONLY"), \
             patch("agent.inspire_diverge.check_bound", return_value=(True, "")):
            result = inspire_diverge(state, model=mock_llm)

        queries = result.get(StateField.INSPIRE_QUERIES, [])
        evidence = result.get(StateField.INSPIRE_EVIDENCE, [])
        if queries != ["角度1", "角度2"]:
            return InspireTestResult(case, False, f"queries 错误: {queries}")
        if len(evidence) != 4:  # 2 queries × 2 tools
            return InspireTestResult(case, False, f"evidence 数量错误: {len(evidence)}")
        return InspireTestResult(case, True, "diverge 写入 queries + evidence ✓")

    if name == "converge_gate_pass":
        draft = "我觉得认知科学里的工作记忆模型，而且可以和 RAG 的 chunk 策略类比。"
        mock_llm = build_mock_llm(
            draft,
            '{"passed": true, "reason": "满足跨域与证据要求"}',
        )
        state = build_base_state(**{
            StateField.INSPIRE_EVIDENCE: [
                ToolResult(
                    success=True,
                    data="认知科学证据",
                    artifacts={"tool": "rag_search", "query": "记忆"},
                ),
            ],
        })

        with patch("agent.inspire_converge.save_message"):
            result = inspire_converge(state, model=mock_llm)

        final = result.get(StateField.FINAL_RESPONSE, "")
        if final != draft:
            return InspireTestResult(case, False, f"final_response 不匹配: {final[:50]}...")
        return InspireTestResult(case, True, "gate 通过，final_response = draft ✓")

    if name == "converge_gate_fail_still_output":
        draft = "我觉得可以换个角度，而且这仍应输出给用户。"
        mock_llm = build_mock_llm(
            draft,
            '{"passed": false, "reason": "缺少跨域连接"}',
        )
        state = build_base_state(**{
            StateField.INSPIRE_EVIDENCE: [],
        })

        with patch("agent.inspire_converge.save_message"):
            result = inspire_converge(state, model=mock_llm)

        final = result.get(StateField.FINAL_RESPONSE, "")
        if final != draft:
            return InspireTestResult(
                case, False,
                "Phase 1：gate 不通过也应输出 draft，实际: " + final[:50],
            )
        return InspireTestResult(case, True, "gate 不通过仍输出 draft ✓")

    return InspireTestResult(case, False, f"未知用例：{name}")


# ── 报告输出 ──────────────────────────────────────────────────────

def print_report(results: list[InspireTestResult]) -> int:
    """打印测试报告，返回失败数量。"""
    print("\n" + "=" * 60)
    print("  Inspire 专项测试 — 8 条 mock 场景")
    print("=" * 60)

    passed = failed = 0
    for r in results:
        tag = "✓ PASSED" if r.passed else "✗ FAILED"
        if r.passed:
            passed += 1
        else:
            failed += 1
        print(f"\n{tag} {r.case.name}")
        print(f"      {r.case.description}")
        print(f"      {r.message}")

    print("\n" + "-" * 60)
    print(f"  通过 {passed} | 失败 {failed}")
    print("=" * 60 + "\n")
    return failed


# ── 入口 ──────────────────────────────────────────────────────────

def main() -> int:
    cases = [
        InspireTestCase("parse_queries_valid", "正常 JSON → 3 条 query"),
        InspireTestCase("parse_queries_fallback", "非法 JSON → 兜底原问题"),
        InspireTestCase("parallel_search_count", "2 query × 2 tool → 4 条 evidence"),
        InspireTestCase("bound_rejected", "BOUND 拒绝单条，其余继续"),
        InspireTestCase("format_evidence_filters", "格式化时过滤失败/空数据"),
        InspireTestCase("diverge_node_state", "inspire_diverge 写入 State"),
        InspireTestCase("converge_gate_pass", "证据门通过 → 输出 draft"),
        InspireTestCase("converge_gate_fail_still_output", "证据门不通过仍输出 draft"),
    ]

    results = [run_inspire_test(c) for c in cases]
    failed_count = print_report(results)
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
