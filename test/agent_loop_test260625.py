"""
阶段 6.4 Loop 专项测试 — 用 mock 覆盖 4 条循环停止场景

运行：
  cd personal_assistant && PYTHONPATH=. uv run python test/agent_loop_test260625.py

设计原则：
  - 不调真实 LLM、不连数据库、不走网络 → 纯 mock，秒级跑完
  - 直接测试 tool_execute 函数（单元测试），不跑完整 Graph
  - 用 unittest.mock.patch 替换 get_tool / get_llm / check_bound / get_tool_bound

覆盖场景：
  1. 结果充分 → 1 轮，enough_information
  2. 弱结果 + LLM CONTINUE → 至少 2 轮
  3. max_turns=1 → reach_max_turns
  4. token_budget_estimate=100 + 大结果 → budget_exceeded
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from unittest.mock import patch, MagicMock

from agent.state import MainState, StateField
from agent.tools.base import ToolResult, ToolContext


# ── 测试结果数据结构 ──────────────────────────────────────────────

@dataclass
class LoopTestCase:
    """单条 loop 测试用例。"""
    name: str                          # 用例名称
    description: str                   # 一句话描述测什么
    expected_stop_reason: str          # 期望的 loop_stop_reason
    expected_min_turns: int = 1        # 期望最少轮数
    expected_max_turns: int = 999      # 期望最多轮数


@dataclass
class LoopTestResult:
    """单条测试的执行结果。"""
    case: LoopTestCase
    passed: bool
    message: str
    actual_stop_reason: str = ""
    actual_turn_count: int = 0


# ── Mock 辅助函数 ─────────────────────────────────────────────────

def build_mock_tool(*return_results: ToolResult):
    """构造一个假工具函数，按顺序返回预设的 ToolResult。

    Args:
        *return_results: 每次调用依次返回的 ToolResult，最后一个会无限重复
            ^-- 不定长参数，让你传任意多个预设结果，如：build_mock_tool（短结果，长结果）传俩
    Returns:
        一个签名为 (ToolContext, str) -> ToolResult 的可调用对象

    示例:
        mock = build_mock_tool(
            ToolResult(success=True, data="短"),
            ToolResult(success=True, data="足够长的数据" * 20),
        )
        # 第 1 次调用 → "短"，第 2 次及以后 → 长数据
    """
    # 闭包状态：用一个列表记录“当前该返回第几个”
    call_count = [0]    #  用单元素列表替代整数，因为闭包里 call_count += 1 会触发
                        # UnboundLocalError（Python 认为你在给局部变量赋值）。call_count[0] += 1
                        # 是修改列表元素，不触发这个规则

    def mock_tool(ctx: ToolContext, question: str) -> ToolResult:
        """假的工具函数，签名和真实工具一摸一样"""
        if not return_results:          # 传 0 个会导致下面的索引错误
            raise ValueError("至少传一个结果")

        # 取当前索引对应的结果，call_count 发生越界等故障（min（999，2））则是超出，用最后一个
        idx = min(call_count[0], len(return_results) - 1)
        result = return_results[idx]
        call_count[0] += 1
        return result

    return mock_tool

def build_mock_llm(*responses: str) -> MagicMock:
    """构造一个假 LLM，按顺序返回预设的文本响应。

    Args:
        *responses: 每次 invoke() 依次返回的 response.content，最后一个会无限重复
            也是预先写好的LLM回复，如responses = ("CONTINUE", "STOP")，两个字符串。
            它们模拟的是：第 1 次问 LLM "信息够不够？" → 它回 "CONTINUE"；第 2 次问 → 它回 "STOP"。

    Returns:
        一个 MagicMock，其 .invoke() 返回值有 .content 属性

    示例:
        mock_llm = build_mock_llm("CONTINUE", "STOP")
        # 第 1 次 invoke → MagicMock(content="CONTINUE")
        # 第 2 次 invoke → MagicMock(content="STOP")
    """
    if not responses:
        raise ValueError("至少传一个响应")
    
    call_count = [0]

    def side_effect(prompt: str) -> str:
        """每次 invoke(prompt) 被调用时，构造一个跟真实invoke返回对象一样的假货"""
        idx = min(call_count[0], len(responses) - 1)
        content = responses[idx]
        call_count[0] += 1
        # MagicMock(content="STOP") -> mock.content 返回 "STOP"
        return MagicMock(content=content)
                #  如果 model.invoke(prompt) 直接返回字符串 "CONTINUE"，那下一行 response.content就会炸——字符串没有 .content 属性。
                #  用 MagicMock 包装一下，假装它有 .content 属性，这样 response.content 就能正常工作了。
    
    mock_llm = MagicMock()
    # side_effect 函数（自定义函数）：从 responses 列表里取一个字符串，包成 MagicMock(content=字符串) 返回。这样 response.content 就能取到值了。
    mock_llm.invoke.side_effect = side_effect
    return mock_llm

def build_base_state(**overrides) -> MainState:
    """构造 tool_execute 所需的最小 MainState，允许覆盖特定字段。

    Args:
        **overrides: 要覆盖的 State 字段，如 target_tool="rag_search", max_turns=1

    Returns:
        一个合法的 MainState 字典，包含所有 tool_execute 需要的字段
    """
    base: MainState = {
        StateField.SESSION_ID: "test-loop-session",
        StateField.USER_QUESTION: "测试问题",
        StateField.TARGET_TOOL: "rag_search",
        # 以下给默认值，避免 tool_execute 里 .get() 时报错 KeyError
        StateField.HISTORY_MESSAGES:[],
        StateField.MEMORY_CONTEXT: "",
        StateField.INTENT: "use_tool",
        StateField.TOOL_RESULTS: [],
        StateField.TURN_COUNT: 0,
        StateField.NEED_CONTINUE: False,
        StateField.FINAL_RESPONSE: "",
        StateField.TRACE_ID: "",
    }

    # overrides 覆盖上面任意字段————比如 max_turns=1
    base.update(overrides)
    return base


# ── 测试用例实现 ──────────────────────────────────────────────────

def run_loop_test(case: LoopTestCase) -> LoopTestResult:
    """执行单条 loop 测试：构造 State → mock 依赖 → 调用 tool_execute → 断言。

    Args:
        case: 测试用例定义

    Returns:
        LoopTestResult 包含通过/失败和实际值
    """
    from agent.main_graph import tool_execute

    # -- 1. 根据用例决定 mock 策略 --
    if case.name =="enough_information":
        # 工具返回 >80 字符 -> _should_continue_by_observation 规则 4
        mock_tool = build_mock_tool(
            ToolResult(success=True, data="A" * 100),
        )
        mock_llm = build_mock_llm("STOP")   # 不会被调用但需要被提供
        state = build_base_state()

    elif case.name == "llm_continue":
        # 第 1 轮工具返回短数据 -> 规则不命中 -> LLM 判 CONTINUE
        # 第 2 轮工具返回长数据 -> 规则 4 命中 enough_information
        mock_tool = build_mock_tool(
            ToolResult(success=True, data="短数据"),    # 第一次调用
            ToolResult(success=True, data="B" * 100),   # 第二次调用
        )
        mock_llm = build_mock_llm("CONTINUE", "STOP")   # 第一次 LLM -> CONTINUE，第二次 LLM -> STOP
        state = build_base_state()

    elif case.name == "reach_max_turns":
        # max_turns=1 -> 规则 1 直接命中，不会调 LLM
        mock_tool = build_mock_tool(
            ToolResult(success=True, data="短数据")
        )
        mock_llm = build_mock_llm("STOP")
        state = build_base_state(max_turns=1)

    elif case.name == "budget_exceeded":
        # token_budget=100，question + data 一次就超
        # question 默认 "测试问题" (4字) + data 100字 = 104 > 100
        mock_tool = build_mock_tool(
            ToolResult(success=True, data="X" * 100),
        )
        mock_llm = build_mock_llm("STOP")
        state = build_base_state(**{StateField.TOKEN_BUDGET_ESTIMATE: 100})

    else:
        return LoopTestResult(
            case=case, passed=False,
            message=f"未知用例：{case.name}"
        )

    # -- 2. 用 patch 替换四个外部依赖
    with patch("agent.main_graph.get_tool", return_value=mock_tool), \
         patch("agent.main_graph.get_llm", return_value=mock_llm), \
         patch("agent.main_graph.check_bound", return_value=(True, "")), \
         patch("agent.main_graph.get_tool_bound", return_value="READ_ONLY"):
        # -- 3. 调用 tool_execute 执行 --
        result_dict = tool_execute(state)

    # -- 4. 断言 --
    actual_reason = result_dict.get(StateField.LOOP_STOP_REASON, "unknown")
    actual_turns = result_dict.get(StateField.TURN_COUNT, 0)

    checks = []
    # 检查停止原因
    if actual_reason == case.expected_stop_reason:
        checks.append(f"stop_reason={actual_reason} √")
    else:
        checks.append(f"stop_reason 期望 {case.expected_stop_reason}，实际 {actual_reason}")

    # 检查轮数范围
    if case.expected_min_turns <= actual_turns <= case.expected_max_turns:
        checks.append(f"turn_count={actual_turns} √")
    else:
        checks.append(f"turn_count 期望 {case.expected_min_turns}-{case.expected_max_turns}，实际 {actual_turns}")

    # 最后统一判断：只要有一条消息含"期望"就是失败了
    failed = [msg for msg in checks if "期望" in msg]
    if failed:
        return LoopTestResult(
            case=case, passed=False,
            message=";".join(failed),
            actual_stop_reason=actual_reason,
            actual_turn_count=actual_turns
        )
    return LoopTestResult(
        case=case, passed=True,
        message=";".join(checks),
        actual_stop_reason=actual_reason,
        actual_turn_count=actual_turns
    )





# ── 报告输出 ──────────────────────────────────────────────────────

def print_report(results: list[LoopTestResult]) -> int:
    """打印测试报告，返回退出码（0=全部通过，1=有失败）。

    Args:
        results: 测试结果列表

    Returns:
        int: 0 表示全部通过，1 表示存在失败
    """
    print("\n" + "=" * 60)
    print("  Loop 专项测试 — 4 条循环停止场景")
    print("=" * 60)

    passed = 0
    failed = 0
    for r in results:
        tag = "√ PASSED" if r.passed else "✗ FAILED"
        if r.passed:
            passed += 1
        else: 
            failed += 1

        print(f"\n{tag} {r.case.name}")
        print(f"      {r.case.description}")
        print(f"      {r.message}")

        if not r.passed:
            # 失败时额外展示实际值，方便定位
            print(f"      实际 stop_reason={r.actual_stop_reason}")
            print(f"      实际轮数={r.actual_turn_count}")
    
    print("\n" + "-" * 60)
    print(f"  通过 {passed} | 失败 {failed}")
    print("=" * 60 + "\n")

    return failed # 返回失败数目，main决定是否退出






# ── 入口 ──────────────────────────────────────────────────────────

def main() -> int:
    """主入口：构建用例 → 逐个执行 → 打印报告 → 返回退出码。"""
    cases = [
        LoopTestCase(           # 数据是否足够？正常查询的最高频场景
            name="enough_information",
            description="工具返回 >80 字符 -> 1 轮即停",
            expected_stop_reason="enough_information",
            expected_min_turns=1,
            expected_max_turns=1,
        ),
        LoopTestCase(           # 验证 LLM 判 CONTINUE 后会继续执行
            name="llm_continue",
            description="短数据 + LLM 判 CONTINUE -> 至少 2 轮",
            expected_stop_reason="enough_information",      # 第 2 轮的长数据出发token超限
            expected_min_turns=2,
            expected_max_turns=2,
        ),
        LoopTestCase(           # 最大循环次数
            name="reach_max_turns",
            description="max_turns=1 -> 1 轮达到上限即停",
            expected_stop_reason="reach_max_turns",
            expected_min_turns=1,
            expected_max_turns=1,
        ),
        LoopTestCase(           # 验证 token 预算超限
            name="budget_exceeded",
            description="token_budget=100 + 大数据 -> 预算超限",
            expected_stop_reason="budget_exceeded",
            expected_min_turns=1,
            expected_max_turns=1,
        ),
    ]

    # 逐个执行
    results = []
    for case in cases:
        result = run_loop_test(case)
        results.append(result)
    
    # 打印报告
    failed_count = print_report(results)
    
    # 退出码：有失败案例 -> 1， 全部通过 -> 0
    return 1 if failed_count > 0 else 0


if __name__ == "__main__":
    # main() 返回 0（通过）或 1（失败），sys.exit(退出码) 把这个数字传给操作系统：
    sys.exit(main())
