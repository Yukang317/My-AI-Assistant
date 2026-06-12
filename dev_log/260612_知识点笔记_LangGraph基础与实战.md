# 知识点笔记：LangGraph 基础概念与 Agent 实战

**日期**：2026-06-12
**来源**：Q&A 对话记录 + 阶段 6.1 Agent 骨架开发实战

---

## 第一部分：LangGraph / TypedDict 基础（Q&A 驱动）

### Q1：TypedDict 是什么？LangGraph 为什么用它？

`TypedDict` 是 Python 的类型注解工具（PEP 589），为字典定义**固定的键名**和**键值类型**。它不是可实例化的类——只是 IDE/类型检查器的提示。

LangGraph 把它用作 **State Schema**（状态蓝图）：
- 声明图中所有节点共享的字段名和类型
- 通过 `Annotated[type, reducer]` 为每个字段指定合并策略
- LangGraph 在构建 StateGraph 时读取 `State.__annotations__` 获取字段信息

### Q2：`__annotations__` 是什么？

`__annotations__` 是 Python 类的特殊属性，以 dict 形式存储所有被注解的字段及其类型。对于 TypedDict，Python 自动生成。LangGraph 通过读取它来识别每个字段的 reducer。

```python
class MainState(TypedDict):
    intent: str
    tool_results: Annotated[list, add]

MainState.__annotations__
# {'intent': str, 'tool_results': typing.Annotated[list, operator.add]}
```

注意：继承时子类 `__annotations__` 会**合并**父类的注解；`__required_keys__` 和 `__optional_keys__` 区分必填/可选字段。

### Q3：`add_messages` 和 `operator.add` 有什么区别？

| Reducer | 行为 | 适用场景 |
|---------|------|----------|
| `add_messages` | 基于消息 `id` 去重：相同 id 则**替换**，不同则**追加** | 对话消息历史（`BaseMessage` 列表） |
| `operator.add` | 简单列表拼接 `old + new`，不去重不替换 | 普通列表（如工具调用结果） |

`add_messages` 是 LangGraph 内置的智能 reducer；`operator.add` 是 Python 标准库的二元函数。在 MainState 中，`tool_results: Annotated[list, add]` 用的是后者——每次工具执行结果**追加**到列表中，不丢历史。

### Q4：TypedDict 能定义在函数内部吗？

可以，这是**惰性求值**技巧。动机：Python 在模块导入时**立即求值**顶层类型注解。如果 TypedDict 在模块顶层且使用了 `Annotated[list, some_reducer]`，则 `some_reducer` 必须在导入时已完全可用——可能因循环导入或初始化顺序导致 `NameError`。

将 TypedDict 定义包裹在函数内，函数只在显式调用时执行，此时所有依赖已加载完毕。但在本项目中我们发现：**LangGraph 的 reducer（`add`、`add_messages`）都是标准 Python 可调用对象，导入时就可用，惰性求值不是必须的**。Move MainState 到模块顶层后一切正常。

### Q5：reducer 是什么？需要自己定义吗？

**reducer** 是 LangGraph 中合并新旧状态值的函数，签名：`(old_value, new_value) -> merged_value`。

- 默认 reducer（无 `Annotated`）是**覆盖**——新值直接替换旧值
- `operator.add` 是 Python 内置实现，执行 `x + y`（对列表即拼接）
- `add_messages` 是 LangGraph 提供的消息专用 reducer
- **不需要自己定义**，除非有特殊合并逻辑（比如去重、取最大值等）

### Q6：`get_main_state_reducer()` 与 `add` 的关系？

**无直接调用或包裹关系**。`get_main_state_reducer()` 只是一个工厂函数，返回一个 TypedDict 类。在该 TypedDict 内部，`tool_results` 字段引用了 `add` 作为 reducer。函数名中的 "reducer" 是社区习惯用语（整个 State 包含 reducer 信息），可理解为 `get_main_state_schema()`。

### Q7：LangGraph 的 State 到底是什么？

State 是图中**所有节点共享的可变字典**。每个节点返回一个 dict（部分状态更新），LangGraph 根据每个字段的 reducer 将返回值合并到全局 State。流程：

```
节点 A return {"intent": "use_tool"}   → State.intent = "use_tool"（覆盖）
节点 B return {"tool_results": [r1]}   → State.tool_results += [r1]（追加）
节点 C return {"tool_results": [r2]}   → State.tool_results += [r2]（追加，不丢 r1）
```

---

## 第二部分：dataclass 与可变默认值陷阱

### ToolContext / ToolResult 设计

```python
@dataclass
class ToolContext:
    session_id: str
    metadata: dict = field(default_factory=dict)  # 每个实例独立空 dict

@dataclass
class ToolResult:
    success: bool                                  # 必填，每次传入新值
    data: str = ""                                 # 有默认值，但 str 不可变
    error: Optional[str] = None                    # None 是全局单例，不可变
    artifacts: dict = field(default_factory=dict)  # 每个实例独立空 dict
```

### 核心陷阱：`artifacts: dict = {}`

如果写成 `artifacts: dict = {}`，Python 在**类定义时**（不是实例化时）创建 **一个** 空 dict，所有实例的 `artifacts` 都指向这个**共享**对象：

```python
r1 = ToolResult(success=True, data="a")
r2 = ToolResult(success=True, data="b")
r1.artifacts["key"] = "v"   # 修改的是共享 dict
print(r2.artifacts)          # {"key": "v"} ← 被污染了！
```

`field(default_factory=dict)` 在**每次实例化时**调用 `dict()` 创建新对象，互不干扰。

### 为什么 `error: Optional[str] = None` 安全？

`None` 是全局唯一的**不可变**单例。当你写 `r1.error = "damage"`，Python 做的事是让 `r1.error` **放弃指向 None，转向指向新字符串**。其他实例的 `error` 仍然指向 `None`，不受影响。

### 为什么 `success: bool` 安全？

没有默认值的字段，实例化时**必须传入值**——每个实例独立存储，不存在共享问题。

### 一句话总结

> 可变默认值（`[]`、`{}`、`set()`）是 Python 的经典陷阱——默认值对象在类定义时创建一次，所有实例共享。不可变默认值（`None`、`0`、`""`）安全。需要可变默认值时用 `field(default_factory=...)`。

---

## 第三部分：阶段 6.1 实战模式（薄封装 · JSON prefill · 防注入 · 检查点）

### 1. 薄封装适配器（Thin Wrapper）

**问题**：LangGraph 节点签名必须是 `(state: TypedDict) -> dict`，但业务函数需要额外参数。

**解决**：闭包捕获额外参数：

```python
def build_graph():
    llm = get_llm()   # 在闭包外层获取一次

    def intent_node(state: MainState) -> dict:
        # 内层函数捕获了外层 llm 变量
        return intent_route(state, llm)

    def result_synthesis_node(state: MainState) -> dict:
        return result_synthesis(state, llm)

    graph.add_node("intent_route", intent_node)
    graph.add_node("result_synthesis", result_synthesis_node)
```

**注意事项**：
- 闭包在 `build_graph()` 内定义，LLM 只初始化一次
- 薄封装只做参数转换，不含业务逻辑——保持薄
- 后续如果想在节点间传更多上下文，在 State 加字段，不要改薄封装

### 2. JSON prefill — 从源头杜绝 LLM markdown 噪声

**问题**：LLM 本能地输出 "好的，这是 JSON：\`\`\`json..."，干扰结构化解析。

**解决**：让 LLM 以为 assistant 已经开始写 JSON：

```python
messages = [
    SystemMessage(content="...指令..."),
    HumanMessage(content=question),
    AIMessage(content="{"),   # LLM 看到 assistant 已输出 "{"，直接续写
]
```

LLM 的续写本能驱使它从 `{` 之后继续输出，跳过寒暄和 ````json` 标记。这是利用 LLM "完成已有文本" 的训练范式。

### 3. 防 Prompt Injection

**核心原则**：外部数据（工具搜索结果、用户上传文件）**绝不**放进 SystemMessage。

- SystemMessage = `role="system"` → LLM 倾向**无条件信任**
- HumanMessage = `role="user"` → LLM 视为 **可质疑的用户数据**

```python
# ✓ 正确：工具结果放 HumanMessage
messages.append(HumanMessage(content=f"{question}\n\n【工具检索结果】\n{results_text}"))

# ✗ 错误：工具结果拼接到 SystemMessage
system_content += f"\n\n{results_text}"  # 外部内容可能包含对抗性指令
```

**同样**：不要用 `+=` 把外部数据拼到 SystemMessage 中——用 `.format()` 占位符注入可控内容（如 memory_context、bound_summary），不可控内容放 HumanMessage。

### 4. MemorySaver 检查点机制

```python
graph = builder.compile(checkpointer=MemorySaver())
```

- 每次节点执行后**自动保存** State 快照
- `config["configurable"]["thread_id"]` → `session_id` 实现多会话隔离
- 同一 `thread_id` 的下次调用从上次快照继续（多轮对话）
- 当前用内存存储（进程重启丢失），生产环境换 `SqliteSaver` 或 `PostgresSaver`

### 5. 工具注册表模式

```python
TOOLS = {
    "rag_search":     {"func": rag_search,     "description": "...", "category": "knowledge"},
    "web_search":     {"func": web_search,     "description": "...", "category": "external"},
    "document_search":{"func": document_search,"description": "...", "category": "files"},
}

def get_tool(name: str) -> Callable:
    return TOOLS[name]["func"]   # 返回的是函数对象，不是完整 dict
```

注册表 = 集中管理工具元数据（名称、描述、分类），`get_tool()` 返回可调用函数。添加新工具只需在 `TOOLS` 中注册一条记录。

---

## 第四部分：Agent 数据流全景

```
用户提问
  │
  ▼
load_context          ← PG 加载历史消息 + MEMORY.md 读取用户画像
  │
  ▼
intent_route          ← LLM JSON prefill 判断意图
  │
  ├─ general_chat ──────────────────────┐
  │                                     │
  └─ use_tool                            │
       │                                 │
       ▼                                 │
  tool_execute        ← 工具注册表       │
       │                                 │
       └─────────────────────────────────┘
                    │
                    ▼
            result_synthesis   ← LLM 合成 + PG 保存
                    │
                    ▼
                   END
```

关键数据流路径：

```
ToolContext(session_id)  →  rag_search(ctx, query)  →  ToolResult(success, data)
       ↑                         ↑                            ↑
  编排层桥接                  PG 上下文                    LangGraph State 传递
```

---

## 附录：阶段 6.1 Bug 速查

| # | 问题 | 根因 | 修复 |
|---|------|------|------|
| 1 | `TypeError: 'dict' object is not callable` | `get_tool()` 返回了完整 TOOLS[name] dict | 改为 `TOOLS[name]["func"]` |
| 2 | `result_synthesis` 签名不兼容 | `(state, model)` 两个参数，LangGraph 只传 state | 添加 `result_synthesis_node` 薄封装 |
| 3 | `rag_search.py` import 重复 | 合并文件时残留 | 删除重复块 |
| 4 | `ImportError: MainState` | TypedDict 定义在函数内部 | 提到模块顶层 |
