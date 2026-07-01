可以，按 **Mode F** 给你一份“三目标一体化”实施方案。你按这套做完后，我再做逐文件验收。

## 实施范围（只动这两个文件）

- `personal_assistant/agent/main_graph.py`
- `personal_assistant/agent/state.py`

---

## 目标 1：while-true 多轮工具循环（max_turns）

### 1) `state.py` 先补循环控制字段（建议新增）
在 `StateField` 增加：
- `MAX_TURNS = "max_turns"`
- `LOOP_STOP_REASON = "loop_stop_reason"`  （如：`"reach_max_turns" | "llm_stop" | "budget_exceeded" | "timeout"`）
- `TOOL_BUDGET_USED = "tool_budget_used"`（先按“调用次数预算”）
- `TIME_BUDGET_MS = "time_budget_ms"`（总超时预算）
- `ELAPSED_MS = "elapsed_ms"`（当前累计耗时）

在 `MainState` 增加对应字段（`NotRequired` 可选也行）：
- `max_turns: int`
- `loop_stop_reason: str`
- `tool_budget_used: int`
- `time_budget_ms: int`
- `elapsed_ms: int`

---

### 2) `main_graph.py` 把 `tool_execute()` 改成循环引擎
你现在 `tool_execute()` 只跑一轮，直接返回。改成：

- 入参读：
  - `target_tool`、`user_question`、`session_id`、`trace_id`
  - `turn_count`（默认 0）
  - `max_turns`（默认 3，先写死常量也行）
- 循环条件：
  - `while turn_count < max_turns:`
- 每轮流程（固定顺序）：
  1. 做 `check_bound`
  2. 执行工具
  3. 记录 trace（你已有 `trace_bound_check`/`trace_tool_execute`）
  4. 更新 `turn_count += 1`
  5. 把 `result` append 到 `tool_results`
  6. 调“继续/停止判定器”（目标2）
  7. 若停止则 break，否则进入下一轮

返回时统一给：
- `tool_results`
- `turn_count`
- `need_continue=False`（离开该节点时统一置 false，避免图重复进节点）
- `loop_stop_reason`

> 注意：Graph 结构先不用改边，仍然是 `tool_execute -> result_synthesis`。循环完全在节点内部完成。

---

## 目标 2：基于工具结果决定是否继续调用

### 3) 新增一个内部判定函数（放 `main_graph.py`）
建议函数签名：

```python
def _should_continue_by_observation(
    question: str,
    latest_result: ToolResult,
    turn_count: int,
    max_turns: int,
    model: BaseChatModel,
) -> tuple[bool, str]:
    ...
```

返回：
- `bool`: 是否继续
- `str`: 原因（用于 `loop_stop_reason`）

### 4) 判定策略（先“规则优先 + LLM 兜底”）
按这个顺序，稳定且容易测：

1. **硬停止**
   - `turn_count >= max_turns` -> stop, `"reach_max_turns"`
2. **工具失败**
   - `latest_result.success is False` -> stop, `"tool_failed"`
3. **结果已充分**
   - 规则：`latest_result.data` 非空且长度超过阈值（比如 > 80） -> stop, `"enough_information"`
4. **不充分才问 LLM**
   - 给 LLM 一个极短 prompt，只输出 `CONTINUE`/`STOP`
   - `CONTINUE` 才继续，否则 stop（`"llm_stop"`）

这样你不会把“是否继续”完全交给 LLM，稳定性更高。

---

## 目标 3：预算控制（token / 次数 / 超时）

### 5) 次数预算（最先落地）
- 预算变量：`max_turns`
- 每轮 `turn_count += 1`
- 命中上限直接停，`loop_stop_reason="reach_max_turns"`

### 6) 超时预算（同一版一起做）
- 在 `tool_execute` 开头：`node_t0 = time.monotonic()`
- 每轮结束更新 `elapsed_ms`
- 若 `elapsed_ms >= time_budget_ms` -> stop, `"timeout"`

### 7) token 预算（先做“软预算”）
你现在不一定能稳定拿到真实 token usage，先做软预算：
- 每轮估算：`len(question) + len(str(latest_result.data))`
- 累加到本地变量 `token_budget_used_estimate`
- 超阈值停：`"budget_exceeded"`

后续再替换为模型/SDK真实 usage 字段。

---

## 你可以直接照抄的改造顺序（推荐）

1. 先改 `state.py` 字段（一次性加全）
2. 再改 `tool_execute()` 成 while 循环（先只做“次数预算”）
3. 加 `_should_continue_by_observation()`（先规则，不接 LLM）
4. 接入超时预算
5. 最后加 LLM 兜底判定（CONTINUE/STOP）

---

## 完成后自测清单（你做完我再验收）

- `uv run python -m py_compile agent/state.py agent/main_graph.py`
- 跑黄金路径测试：
  - `uv run python test/agent_paths_test.py`
- 手工测 3 条：
  - 普通一次工具就够 -> 1 轮停止
  - 工具返回空/弱结果 -> 至少 2 轮
  - 故意设置 `max_turns=1` -> 必然因上限停止

---

你做完后把这两个文件给我，我会按 **Mode F 验收标准** 检查：
- 字段是否齐全
- while 循环是否真生效
- 继续/停止判定是否按“规则优先”
- 预算是否都能触发并写入 `loop_stop_reason`
- 是否破坏现有 `trace` 与 `result_synthesis` 链路。