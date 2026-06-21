from agent.main_graph import run_graph_debug

state = run_graph_debug(
    'memory-test-001',
    '我在做 personal_assistant 个人 AI 助理项目，目标是找工作。'
    '我喜欢小步教学、不说废话、直接上手，讨厌说教和啰嗦。'
)

print("=== 最终回复 ===")
result = state.get('final_response', '')
print(result)
print("-"*50)
print("=== State 里的 memory_context 长度 ===")
print(len(state.get('memory_context', '')), "字符")