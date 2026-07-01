# 代码审阅：main_graph.py + state.py 问题分析与修复指南

> 日期：2026-06-25
> 对比基准：`personal_assistant/ref_loop.md`（三目标一体化方案）

---

## 总体评价

你已经搭出了骨架，不是"完全不对"——`StateField` 字段基本齐全、while 循环的轮廓在、`_should_continue_by_observation` 函数也有雏形。但有几个**关键 bug** 导致循环逻辑没跑通，下面是逐文件逐问题的分析。

---

## 一、`state.py` 的问题

### 问题 1：`history_messages` 类型太模糊

**当前代码（第 82 行）：**
```python
history_messages: list       # 最近 20 条历史消息
```

**问题：** `list` 没指定元素类型，LangGraph 不知道该怎么合并多轮消息。而且你已经 import 了 `add_messages` 和 `BaseMessage`（第 18-19 行），只是没用上。

**修复：**
```python
history_messages: Annotated[list[BaseMessage], add_messages]
```

这是 sagt_agent 的标准做法——`add_messages` reducer 保证新消息追加而不是覆盖旧消息。

---

### 问题 2：循环控制字段没标 `NotRequired`

**当前代码（第 101-105 行）：**
```python
# 关于 loop
max_turns: int              # 最大循环次数
loop_stop_reason: str       # 循环停止原因
tool_budget_used: int       # 调用次数运算
time_budget_ms: int         # 总超时预算
elapsed_ms: int             # 当前累计耗时
```

**问题：** `ref_loop.md` 明确说这些字段应该是 `NotRequired`（可选）。目前是必填字段——如果调用方（比如 `_invoke_with_trace`）不给这些字段，类型检查就过不了。

**修复：**
```python
max_turns: NotRequired[int]
loop_stop_reason: NotRequired[str]
tool_budget_used: NotRequired[int]
time_budget_used: NotRequired[int]   # 注意：ref_loop.md 里叫 TIME_BUDGET_MS
elapsed_ms: NotRequired[int]
```

`NotRequired` 你已经从 `typing` import 了（第 11 行），直接用就行。

---

## 二、`main_graph.py` 的问题（重灾区）

### 🔴 致命 Bug 1：`_should_continue_by_observation` 从未被调用

你在第 70-99 行定义了这个函数，但 while 循环里（第 138-208 行）**一次都没调它**。

当前 while 循环的实际流程：
```
安检 → trace_bound_check → turn_count += 1 → 执行工具 → trace_tool_execute → 回到 while 开头
```

**没有判定器，没有 break。** 循环会傻傻跑满 `max_turns` 次，每次都调同一个工具，完全不管工具结果是否已经够用。

`ref_loop.md` 要求的流程：
```
安检 → 执行工具 → trace → turn_count += 1 → append 结果 → 调判定器 → break 或 continue
```

---

### 🔴 致命 Bug 2：`model.get_response()` 不存在

**当前代码（第 96 行）：**
```python
if model.get_response(question, latest_tool_result.data) == "继续调用":
```

LangChain 的 `BaseChatModel` **没有** `get_response` 方法。这行代码会在运行时直接抛 `AttributeError`。

**正确做法：** 用 `model.invoke()` 发一个短 prompt，让 LLM 输出 `CONTINUE` 或 `STOP`。

---

### 🔴 致命 Bug 3：`turn_count` 递增时机不对

**当前代码（第 161 行）：** 在安检通过后、执行工具**之前**就把 `turn_count += 1`：

```python
# H2：更新循环次数
turn_count += 1        # ← 太早了！

if not allowed:
    return {           # 安检失败时 turn_count 已经 +1 了
        ...
        StateField.TOOL_BUDGET_USED: turn_count,  # ← 比实际调用次数多 1
    }

# ── 第 2 步：安检通过，真正执行工具 ──
result = tool_func(ctx, question)   # ← 工具还没执行，turn_count 已经是 1 了
```

**正确时机：** 工具执行完毕 + trace 记录完毕后，再 `turn_count += 1`。

---

### 🟡 Bug 4：结果没有累积（只保留最后一轮）

**当前代码（第 210 行）：**
```python
return {
    StateField.TOOL_RESULTS: [result],   # ← 永远只有一个元素
```

while 循环里每轮都会**覆盖** `result` 变量，最后 return 时只返回最后一轮的结果。前几轮的搜索结果全部丢失。

**修复：** 在循环开始前初始化 `tool_results_list = []`，每轮 `append(result)`，最后一起返回。

---

### 🟡 Bug 5：`elapsed_ms` 没有累积

**当前代码（第 199 行）：**
```python
elapsed_ms = int((time.monotonic() - t0) * 1000)   # 每轮覆盖
```

`t0` 是**单轮**工具执行的起点（第 189 行），所以 `elapsed_ms` 只反映最后一轮的耗时。前几轮的时间被丢弃了。

**修复：** 在循环前加 `node_t0 = time.monotonic()`，需要累计耗时时用 `int((time.monotonic() - node_t0) * 1000)`。

---

### 🟡 Bug 6：循环结束后的 return 全是硬编码值

**当前代码（第 210-216 行）：**
```python
return {
    StateField.TOOL_RESULTS: [result],
    StateField.NEED_CONTINUE: False,
    StateField.LOOP_STOP_REASON: "工具调用成功",  # ← 永远写这个，不管实际情况
    StateField.TOOL_BUDGET_USED: turn_count,
    StateField.ELAPSED_MS: elapsed_ms,
}
```

不论循环是因为「达到上限」「工具失败」「超时」还是「LLM 判定够了」而结束，`loop_stop_reason` 永远是硬编码的 `"工具调用成功"`。后续 debug 时你完全不知道循环为什么停的。

**修复：** 用一个 `stop_reason` 变量跟踪真实停止原因，循环结束后写入返回值。

---

### 🟡 Bug 7：缺少超时预算和结果长度规则

`ref_loop.md` 目标 3 要求的判定策略（规则优先）：

| 优先级 | 条件 | 动作 |
|--------|------|------|
| 1 | `turn_count >= max_turns` | stop, `"reach_max_turns"` |
| 2 | `result.success is False` | stop, `"tool_failed"` |
| 3 | `elapsed_ms >= time_budget_ms` | stop, `"timeout"` |
| 4 | `len(data_str) > 80` | stop, `"enough_information"` |
| 5 | 前 4 条都不触发 | 问 LLM：CONTINUE 或 STOP |

你的 `_should_continue_by_observation` 只做了 1、2、5（且 5 的实现是坏的）。缺少 3 和 4。

---

## 三、修复参考代码

### 第一步：修 `_should_continue_by_observation`

```python
def _should_continue_by_observation(
    question: str,
    latest_result: ToolResult,
    turn_count: int,
    max_turns: int,
    elapsed_ms: int,
    time_budget_ms: int,
    model: BaseChatModel,
) -> tuple[bool, str]:
    """规则优先 + LLM 兜底：判断是否需要继续调用工具。

    五层判定（按优先级）：
      1. turn_count >= max_turns  → stop (reach_max_turns)
      2. 工具失败                  → stop (tool_failed)
      3. 超时                      → stop (timeout)
      4. 结果已充分（>80字符）      → stop (enough_information)
      5. LLM 兜底                  → CONTINUE 则继续，否则 stop (llm_stop)

    Args:
        question: 用户原始问题
        latest_result: 最新一轮工具执行结果
        turn_count: 当前已执行轮数
        max_turns: 最大允许轮数
        elapsed_ms: 本节点已耗时（毫秒）
        time_budget_ms: 超时预算（毫秒）
        model: LLM 实例

    Returns:
        (是否继续, 停止原因字符串)
    """
    # ── 规则 1：达到最大次数 ──
    if turn_count >= max_turns:
        return False, "reach_max_turns"

    # ── 规则 2：工具执行失败 ──
    if not latest_result.success:
        return False, "tool_failed"

    # ── 规则 3：超时 ──
    if elapsed_ms >= time_budget_ms:
        return False, "timeout"

    # ── 规则 4：结果已充分 ──
    data_str = str(latest_result.data) if latest_result.data else ""
    if len(data_str) > 80:
        return False, "enough_information"

    # ── 规则 5：LLM 兜底 ──
    prompt = (
        f"用户问题：{question}\n"
        f"工具返回内容：{data_str[:500]}\n\n"
        f"以上信息是否足够回答用户问题？\n"
        f"如果足够，只回复 STOP。如果还需要更多信息，只回复 CONTINUE。\n"
        f"不要解释，不要任何其他文字。"
    )
    try:
        response = model.invoke(prompt)
        # 兼容不同的返回值格式
        if hasattr(response, "content"):
            answer = response.content
        else:
            answer = str(response)
        answer = answer.strip().upper()
    except Exception:
        # LLM 调用失败，保守处理：不继续
        return False, "llm_error"

    if "CONTINUE" in answer:
        return True, "llm_continue"
    return False, "llm_stop"
```

---

### 第二步：修 `tool_execute` 的 while 循环体

完整参考实现（关键改动已用 `★` 标注）：

```python
def tool_execute(state: MainState) -> dict:
    """执行路由选中的工具，while-true 循环直到满足停止条件。

    循环内每轮严格按 ref_loop.md 顺序：
      安检 → 执行工具 → trace → turn_count++ → append 结果 → 判定器 → break?

    Args:
        state: 当前 MainState

    Returns:
        dict: 含 tool_results（累计列表）、turn_count、loop_stop_reason、elapsed_ms
    """
    target_tool = state.get(StateField.TARGET_TOOL, "")
    question = state.get(StateField.USER_QUESTION, "")
    session_id = state.get(StateField.SESSION_ID, "")
    trace_id = state.get(StateField.TRACE_ID, "")

    # ★ 循环控制变量（从 State 取，给默认值）
    max_turns = state.get(StateField.MAX_TURNS, 3)
    time_budget_ms = state.get(StateField.TIME_BUDGET_MS, 30000)

    # ★ 初始化：结果列表 + 计时起点
    tool_results_list: list[ToolResult] = []
    turn_count = 0
    node_t0 = time.monotonic()
    stop_reason = "unknown"
    model = get_llm()

    # ── while-true 循环 ──
    while turn_count < max_turns:

        # ── 第 1 步：Harness 安检（确定性代码，不信任 LLM）──
        try:
            bound_category = get_tool_bound(target_tool)
            action_desc = BOUND_ACTION_MAP.get(bound_category, "未知操作")
            action = f"调用工具:{target_tool} {action_desc}"
            allowed, reason = check_bound(action, question)

            if trace_id:
                trace_bound_check(
                    trace_id, session_id,
                    target_tool=target_tool,
                    allowed=allowed, reason=reason,
                )

            if not allowed:
                stop_reason = "安检失败"
                tool_results_list.append(
                    ToolResult(success=False, data="", error=reason)
                )
                break  # ★ break 而非 return，统一走底部返回值逻辑

        except Exception as e:
            if trace_id:
                trace_bound_check(
                    trace_id, session_id,
                    target_tool=target_tool,
                    allowed=False, reason=str(e),
                )
            stop_reason = "工具名不存在等异常"
            tool_results_list.append(
                ToolResult(success=False, data="", error=str(e))
            )
            break

        # ── 第 2 步：执行工具 ──
        t0 = time.monotonic()
        try:
            tool_func = get_tool(target_tool)
            ctx = ToolContext(session_id=session_id)
            result = tool_func(ctx, question)
        except Exception as e:
            result = ToolResult(success=False, data="", error=str(e))

        step_ms = int((time.monotonic() - t0) * 1000)

        # ★ 第 3 步：累积结果（不是覆盖）
        tool_results_list.append(result)

        # ── 第 4 步：trace ──
        if trace_id:
            trace_tool_execute(
                trace_id, session_id,
                target_tool=target_tool,
                success=result.success,
                latency_ms=step_ms,
                error=result.error or "",
            )

        # ★ 第 5 步：turn_count += 1（放在工具执行之后）
        turn_count += 1

        # ── 第 6 步：继续/停止判定器 ──
        total_elapsed = int((time.monotonic() - node_t0) * 1000)
        should_continue, reason_str = _should_continue_by_observation(
            question=question,
            latest_result=result,
            turn_count=turn_count,
            max_turns=max_turns,
            elapsed_ms=total_elapsed,
            time_budget_ms=time_budget_ms,
            model=model,
        )

        if not should_continue:
            stop_reason = reason_str
            break

    # ── 统一返回（循环正常结束 → 兜底为 reach_max_turns）──
    if stop_reason == "unknown":
        stop_reason = "reach_max_turns"

    total_elapsed_ms = int((time.monotonic() - node_t0) * 1000)

    return {
        StateField.TOOL_RESULTS: tool_results_list,
        StateField.NEED_CONTINUE: False,
        StateField.TURN_COUNT: turn_count,
        StateField.LOOP_STOP_REASON: stop_reason,
        StateField.TOOL_BUDGET_USED: turn_count,
        StateField.ELAPSED_MS: total_elapsed_ms,
    }
```

---

## 四、修复顺序（建议严格按此来）

| 顺序 | 文件 | 改什么 |
|------|------|--------|
| 1 | `state.py` | `history_messages` → `Annotated[list[BaseMessage], add_messages]`；loop 字段加 `NotRequired` |
| 2 | `main_graph.py` | 删除旧的 `_should_continue_by_observation`，换成上面"第一步"的版本 |
| 3 | `main_graph.py` | 删除旧的 `tool_execute`，换成上面"第二步"的版本 |

---

## 五、改完后自测

```bash
# 1. 语法检查
cd personal_assistant && uv run python -m py_compile agent/state.py agent/main_graph.py

# 2. 黄金路径测试（如果有的话）
uv run python test/agent_paths_test.py

# 3. 手工验证：启动 app 后测 3 条
#    - 普通一次工具就够 → 1 轮停止，loop_stop_reason = "enough_information"
#    - 工具返回空/弱结果 → 至少 2 轮，看 turn_count > 1
#    - 故意设 max_turns=1 → 必然因上限停止，loop_stop_reason = "reach_max_turns"
```

---

## 六、核心要点速查

| 要点 | 为什么重要 |
|------|-----------|
| 判定器必须在循环内调用 | 否则循环永远跑满 max_turns，浪费 LLM 调用 |
| `turn_count++` 放在执行后 | 保证计数反映的是「已完成」的轮数，不是「即将执行」 |
| 结果用 `append` 而非覆盖 | 前几轮的搜索结果对 result_synthesis 有价值，不能丢 |
| `stop_reason` 用变量追踪 | 后续 debug 时你能从 JSONL 里看到循环为什么停的 |
| 规则优先于 LLM | 确定性规则（次数/超时/结果长度）比 LLM 判断更稳定、更快、更便宜 |
| `break` 而非 `return` | 统一返回值逻辑，避免安检分支和正常退出分支各写一套 return |
