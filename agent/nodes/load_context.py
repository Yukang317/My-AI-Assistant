"""
上下文加载节点 — 加载历史消息 + 读取 MEMORY.md

在 Graph 中作为第一个节点，为后续节点准备上下文数据。
流：PG 加载最近 20 条历史消息 → 读取 MEMORY.md 文件 → 组装到 State。
"""

from agent.state import MainState


# TODO(human): 实现 load_context(state: MainState) -> dict 函数
# 说明：
#   1. 从 MainState 中取出 session_id（用 StateField.SESSION_ID）
#   2. 调用 db.load_messages(session_id) 加载历史消息（已在 db.py 中实现）
#   3. 只取最近 20 条（MAX_HISTORY = 20），避免上下文窗口过大
#   4. 尝试读取 /root/assistant/personal_assistant/memory/MEMORY.md 文件
#      - 文件存在 → 读取内容作为 memory_context 字符串
#      - 文件不存在 → memory_context 设为 ""（优雅降级）
#   5. 返回 dict：
#      - StateField.HISTORY_MESSAGES → 消息列表
#      - StateField.MEMORY_CONTEXT → MEMORY.md 内容或 ""
#   6. 导入 os 模块检查文件存在，导入 db 模块（from db import load_messages）
#
#   注意：state 是 TypedDict，取值用 state.get(StateField.XXX)，不是 state["xxx"]
#   如果 TypedDict 不够灵活，可以在前面用 MainState = Annotated[dict, ...] 模式
