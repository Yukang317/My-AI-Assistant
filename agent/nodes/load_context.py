"""
上下文加载节点 — 加载历史消息 + 读取 MEMORY.md

在 Graph 中作为第一个节点，为后续节点准备上下文数据。
流：PG 加载最近 20 条历史消息 → 读取 MEMORY.md 文件 → 组装到 State。
"""

from agent.state import MainState, StateField
import os
from db import load_messages as db_load_messages
from langchain_core.messages import HumanMessage, AIMessage

MAX_HISTORY = 20
MEMORY_PATH = "/root/assistant/personal_assistant/memory/MEMORY.md"

def load_context(state: MainState) -> dict:
    """加载会话历史 + 用户画像，为后续节点准备上下文。

    LangGraph 的第一个节点，从 PG 读历史消息、从文件系统读 MEMORY.md，
    把结果写入 State 的 history_messages 和 memory_context 字段。

    Args:
        state: 当前 MainState（TypedDict），至少包含 session_id

    Returns:
        dict: {StateField.HISTORY_MESSAGES: [...], StateField.MEMORY_CONTEXT: "..."}
    """
    session_id = state.get(StateField.SESSION_ID, "")

    # 1. 加载历史消息（仅取最近 20 条），转成 LangChain 消息类型
    all_messages = db_load_messages(session_id)         # list[dict]，按时间升序
    recent = all_messages[-MAX_HISTORY:] if len(all_messages) > MAX_HISTORY else all_messages

    # 把 PG 返回的 dict 转成 LangChain 消息对象（HumanMessage / AIMessage）
    # 这样下游节点（result_synthesis 等）看到的就是统一类型，不会出现 subscriptable 错误
    langchain_messages = []
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        # 忽略其他 role 类型

    # 2. 读取 MEMORY.MD （不存在时优雅降级）
    memory_context = ""
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            memory_context=f.read()

    # 3.返回部分 State 更新————LangGraph 自动 merge
    return {
        StateField.HISTORY_MESSAGES: langchain_messages,
        StateField.MEMORY_CONTEXT: memory_context,
    }
