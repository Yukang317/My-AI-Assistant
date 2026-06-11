"""
结果合成节点 — 整合工具结果 + LLM 生成最终回复 + 保存消息

所有执行路径最终汇聚到这个节点：
- 普通闲聊：直接用 LLM 生成回复
- 工具调用：把工具结果注入 prompt，让 LLM 基于结果生成回复
- 保存用户消息和 AI 回复到 PG（与现有 step23_rag.py 机制一致）
"""

from agent.state import MainState


# ── System Prompt ─────────────────────────────────────────────

# TODO(human): 定义 SYNTHESIS_SYSTEM_PROMPT: str 常量
# 说明：
#   1. 提示词指导 LLM 如何整合工具结果生成回答
#   2. 引用 IRON_LAWS（从 agent.bound 导入 get_bound_summary）
#   3. 有工具结果时：基于结果回答，不要编造信息
#   4. 无工具结果时：正常闲聊模式
#   5. 回答应简洁、准确、用中文


# ── 合成函数 ──────────────────────────────────────────────────

# TODO(human): 实现 result_synthesis(state: MainState, model: BaseChatModel) -> dict 函数
# 说明：
#   1. 从 State 中取出 user_question, tool_results, memory_context, history_messages
#   2. 构造 messages 列表：
#      - SystemMessage（SYNTHESIS_SYSTEM_PROMPT + memory_context + BOUND 约束摘要）
#      - history_messages（历史对话，用 langchain_core.messages 类型）
#      - HumanMessage（用户问题，如果有工具结果则附带 tool_results）
#   3. 调用 model.invoke(messages) 获取 LLM 回复，提取 .content 文本
#   4. 调用 db.save_message(session_id, "user", question) 保存用户消息
#   5. 调用 db.save_message(session_id, "assistant", response) 保存 AI 回复
#   6. 返回 dict：
#      - StateField.FINAL_RESPONSE → AI 回复字符串
#      - "messages" → [AIMessage(content=response)] （LangGraph 流式输出的约定）
#   7. 需要导入：
#      - from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
#      - from db import save_message
#      - from agent.bound import get_bound_summary
#
#   安全注意：所有用户输入都要用 HumanMessage 包装，不拼接到 prompt 模板（防注入）
