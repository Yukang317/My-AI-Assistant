"""
结果合成节点 — 整合工具结果 + LLM 生成最终回复 + 保存消息

所有执行路径最终汇聚到这个节点：
- 普通闲聊：直接用 LLM 生成回复
- 工具调用：把工具结果注入 prompt，让 LLM 基于结果生成回复
- 保存用户消息和 AI 回复到 PG（与现有 step23_rag.py 机制一致）
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from db import save_message
from agent.bound import get_bound_summary
from agent.state import MainState, StateField

# ── System Prompt ─────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """你是一个个人 AI 助理，负责根据上下文和工具结果生成最终回复。

## 你的用户画像
{memory_context}

## 回复规则
1. 如果有工具检索结果，请基于检索结果回答，不要编造信息
2. 如果没有工具结果，以友好的方式直接回复用户
3. 回答应简洁、准确、用中文
4. 不确定时坦诚告知，反问用户获取更多信息
5. 不暴露内部实现细节（如"我调用了RAG搜索"）

{bound_summary}
"""


# ── 合成函数 ──────────────────────────────────────────────────

def result_synthesis(state: MainState, model: BaseChatModel) -> dict:
    """整合工具结果 + LLM 生成最终回复 + 保存消息到 PG。
    
    所有执行路径的汇聚节点：
    - 无工具结果 → 纯 LLM 闲聊回复
    - 有工具结果 → 把结果注入 prompt 让 LLM 基于检索内容回答
    
    Args:
        state: 当前 MainState，包含用户问题、上下文、工具结果等
        model: LangChain ChatModel 实例
    
    Returns:
        dict: {FINAL_RESPONSE: str, "messages": [AIMessage]}
    """
    # 1. 从 State 取数据
    session_id = state.get(StateField.SESSION_ID, "")
    question = state.get(StateField.USER_QUESTION, "")
    tool_results = state.get(StateField.TOOL_RESULTS, [])
    memory_context = state.get(StateField.MEMORY_CONTEXT, "")
    history = state.get(StateField.HISTORY_MESSAGES, [])

    # 2. 构造 SystemMessage — 所有内容通过占位符注入，不做字符串拼接
    system_content = SYNTHESIS_SYSTEM_PROMPT.format(
        memory_context=memory_context,
        bound_summary=get_bound_summary(),
    )

    messages = [SystemMessage(content=system_content)]

    # 3. 在 SystemMessage 中追加历史消息（转换成 LangChain 消息类型）
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # 4. 构造用户当前问题（有工具结果时附带）
    user_content = question
    if tool_results:
        # 工具结果注入 HumanMessage 而非 SystemMessage————防止 prompt injection
        results_text ="\n\n【工具检索结果】\n"
        for tr in tool_results:
            if tr.success:
                results_text += f"{tr.data}\n"
            else:
                results_text += f"（搜索失败：{tr.error}）\n"
        user_content = f"{question}\n{results_text}"
    
    messages.append(HumanMessage(content=user_content))

    # 5. 调用 LLM 生成回复
    response = model.invoke(messages)
    answer = response.content.strip()

    # 6. 保存消息到 PG （先存用户问题，后存 AI 回复）
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

    # 7. 返回 State 更新
    return {
        StateField.FINAL_RESPONSE: answer,          # 系统业务字段，追加state的消息历史
        "messages": [AIMessage(content=answer)],    # LangGraph 流式输出约定，让框架知道这条回复是发给用户的
    }


# ═══════════════════════════════════════════════════════════════════════════════
# messages 拼装示例 — 最终发给 LLM 的列表长什么样
# ═══════════════════════════════════════════════════════════════════════════════
#
# 假设：用户画像为空、之前聊过 2 句、当前问 "RTX 5090多少钱"、RAG 有结果
#
# messages = [
#     # ── 0. 系统指令（占位符一次性注入，不做字符串拼接） ──
#     SystemMessage(content="""你是一个个人 AI 助理...
#
#     ## 你的用户画像
#     （暂无）
#
#     ## 回复规则
#     1. 如果有工具检索结果...
#     ...
#     【系统约束 — 必须遵守】
#     1. 禁止触碰：/etc/*、~/.ssh/*、...
#     2. 禁止动作：删除文件、...
#     3. 铁律：回复必须使用中文、..."""),
#
#     # ── 1~2. 历史对话（从 PG 读出来，转成 LangChain 消息类型） ──
#     HumanMessage(content="你好"),
#     AIMessage(content="你好！有什么可以帮你的？"),
#
#     # ── 3. 当前问题 + 工具检索结果（放 HumanMessage，防 prompt injection） ──
#     HumanMessage(content="""RTX 5090多少钱
#
#     【工具检索结果】
#     RTX 5090 建议零售价 1999 美元，国行 16499 元起..."""),
# ]
#
# ── 序列化后实际发给 LLM API（OpenAI 格式） ──
# [
#     {"role": "system",    "content": "你是一个个人 AI 助理..."},
#     {"role": "user",      "content": "你好"},
#     {"role": "assistant", "content": "你好！有什么可以帮你的？"},
#     {"role": "user",      "content": "RTX 5090多少钱\n\n【工具检索结果】\nRTX 5090..."},
# ]
#
# 关键约定：
# - SystemMessage  → role="system"     → 规则层，定义 AI 怎么说话
# - HumanMessage   → role="user"       → 数据层，用户输入 + 外部检索结果
# - AIMessage      → role="assistant"  → 历史层，之前的 AI 回复
# ═══════════════════════════════════════════════════════════════════════════════


