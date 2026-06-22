"""
阶段 6.3 H3：5 条黄金路径回归测试

运行：
  cd personal_assistant && uv run python test/agent_paths_test.py

注意：
  - 用例会调 LLM + 可能调 Tavily/RAG，需要 Docker 四容器 + .env API Key
  - 路由由 LLM 判定，偶有误判；case 5 标记为 known_issue（H4 输入）
  - Tavily 没 key 时 case 3 自动 skip
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

from agent.main_graph import run_graph_debug
from agent.nodes.memory_update import MEMORY_PATH
from agent.state import StateField


# ── 用例定义 ──────────────────────────────────────────────────

@dataclass
class GoldenCase:
    """单条黄金路径用例。"""

    name: str                # 用例名
    question: str             # 用户问题
    session_id: str = ""      # 会话 ID，默认空串
    expect_intent: str | None = None  # 期望的意图
    expect_tool: str | None = None    # 期望的工具
    expect_no_tool: bool = False     # 期望没有工具调用
    memory_keyword: str | None = None   # 跑完后检查 MEMORY.md 是否含此关键词
    skip: bool = False              # 是否跳过
    skip_reason: str = ""           # 跳过原因
    known_issue: bool = False       # True = 失败记入报告但不计 exit 1（留给 H4）



@dataclass
class CaseResult:
    case: GoldenCase                # 用例对象
    passed: bool                    # 是否通过
    message: str                    # 结果描述
    state: dict = field(default_factory=dict)  # 最终状态


# ── 断言辅助 ──────────────────────────────────────────────────

def assert_intent(state: dict, expected: str) -> tuple[bool, str]:
    actual = state.get(StateField.INTENT, "")
    if actual == expected:
        return True, f"intent={actual} ✓"
    return False, f"intent 期望 {expected}，实际 {actual}"

def assert_tool(state: dict, expected: str) -> tuple[bool, str]:
    actual = state.get(StateField.TARGET_TOOL, "")
    if actual == expected:
        return True, f"tool={actual} ✓"
    return False, f"tool 期望 {expected}，实际 {actual or '(无)'}"

def assert_no_tool(state: dict) -> tuple[bool, str]:
    actual = state.get(StateField.TARGET_TOOL, "")
    # general_chat 的闲聊模式中target_tool 为 ""，不是None
    if len(actual) == 0:
        return True, "未调用工具 ✓"
    return False, f"期望没调用工具，实际调用了 {actual}"

def assert_memory_contains(keyword: str) -> tuple[bool, str]:
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return False, f"无法读取 MEMORY.md: {e}"
    if keyword in content:
        return True, f"MEMORY.md 含「{keyword}」✓"
    return False, f"MEMORY.md 未找到「{keyword}」"

# ── 用例执行 ──────────────────────────────────────────────────

def run_golden_case(case: GoldenCase) -> CaseResult:
    if case.skip:
        return CaseResult(case=case, passed=True, message=f"SKIP: {case.skip_reason}")

    # 每个用例独立 session，避免 checkpointer 串台
    session_id = case.session_id or f"golden-{uuid.uuid4().hex[:8]}"
    state = run_graph_debug(session_id, case.question)

    checks: list[tuple[bool, str]] = []

    if case.expect_intent:
        checks.append(assert_intent(state, case.expect_intent))
    if case.expect_tool:
        checks.append(assert_tool(state, case.expect_tool))
    if case.expect_no_tool:
        checks.append(assert_no_tool(state))
    if case.memory_keyword:
        checks.append(assert_memory_contains(case.memory_keyword))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        return CaseResult(
            case=case,
            passed=False,
            message="; ".join(failed),
            state=state,
        )
    return CaseResult(
        case=case,
        passed=True,
        message="; ".join(msg for _, msg in checks),
        state=state,
    )


# ── 用例构建 ──────────────────────────────────────────────────

def build_golden_cases() -> list[GoldenCase]:
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()

    return [
        GoldenCase(
            name="1-闲聊",
            question="你好，请做一下自我介绍",
            expect_intent="general_chat",
            expect_no_tool=True,
        ),
        GoldenCase(
            name="2-知识库检索",
            question="知识库里有什么关于Python的内容？",
            expect_intent="use_tool",
            expect_tool="rag_search",
        ),
        GoldenCase(
            name="3-实时新闻",
            question="今天有什么科技新闻？",
            expect_intent="use_tool",
            expect_tool="tavily_search",
            skip=not tavily_key,
            skip_reason="TAVILY_API_KEY 未配置",
        ),
        GoldenCase(
            name="4-记忆写入",
            question="我叫 Yukang，喜欢小步教学、讨厌说教",
            session_id=f"golden-mem-write-{uuid.uuid4().hex[:8]}",
            memory_keyword="小步教学",
        ),
        GoldenCase(
            name="5-记忆读取",
            question="我的沟通偏好是什么？",
            session_id=f"golden-mem-read-{uuid.uuid4().hex[:8]}",
            expect_intent="general_chat",
            expect_no_tool=True,
            known_issue=True,  # 路由器可能误判 rag_search，H4 修
        ),
    ]


# ── 打印报告 ──────────────────────────────────────────────────

def print_report(results: list[CaseResult]) -> None:
    print("\n" + "=" * 60)
    print("  Agent 黄金路径回归测试")
    print("=" * 60)

    hard_fail = 0
    known_fail = 0
    passed = 0
    skipped = 0

    print(f"共 {len(results)} 个用例，成功 {sum(1 for r in results if r.passed)} 个")
    for r in results:
        tag = "✅ PASS"
        if r.message.startswith("SKIP:"):
            tag = "⏭️  SKIP"
            skipped += 1
        elif r.passed:
            passed += 1
        elif r.case.known_issue:
            tag = "⚠️  KNOWN"
            known_fail += 1
        else:
            tag = "❌ FAIL"
            hard_fail += 1

        print(f"\n{tag}  {r.case.name}")
        print(f"     {r.message}")
        if not r.passed and r.state:
            print(f"     intent={r.state.get('intent')} tool={r.state.get('target_tool')}")

    print("\n" + "-" * 60)
    print(f"  通过 {passed} | 跳过 {skipped} | 已知问题 {known_fail} | 失败 {hard_fail}")
    print("=" * 60)

    return hard_fail

def main() -> int:
    results = [run_golden_case(c) for c in build_golden_cases()]
    print_report(results)
    # 只有非 known_issue 的失败才返回 1
    hard_fail = sum(
        1 for r in results
        if not r.passed and not r.message.startswith("SKIP:") and not r.case.known_issue
    )
    return 1 if hard_fail else 0

if __name__ == "__main__":
    raise SystemExit(main())