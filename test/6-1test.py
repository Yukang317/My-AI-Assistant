from agent.main_graph import run_graph_debug

def print_state_summary(state: dict, label: str):
    """打印 State 中关键字段的摘要"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  意图 (intent)      : {state.get('intent', '(未设置)')}")
    print(f"  目标工具 (target)  : {state.get('target_tool', '(无)') or '(无)'}")
    print(f"  工具结果数         : {len(state.get('tool_results', []))}")
    print(f"  历史消息数         : {len(state.get('history_messages', []))}")
    print(f"  记忆上下文长度     : {len(state.get('memory_context', ''))} 字符")
    print(f"  循环次数 (turn)    : {state.get('turn_count', '(未设置)')}")
    print(f"  需要继续 (continue): {state.get('need_continue', '(未设置)')}")
    print(f"{'-'*60}")
    final = state.get('final_response', '')
    print(f"  最终回复 ({len(final)} 字符):")
    print(f"  {final[:400]}{'...' if len(final) > 400 else ''}")
    print(f"{'='*60}")

# ── 测试1：闲聊路径 ──
print_state_summary(
    run_graph_debug('session1', '你好，请做一下自我介绍'),
    '测试1: 闲聊路径'
)

# ── 测试2：工具调用路径（具体的问题更易触发 use_tool）──
# 之前的问题是：“知识库里有内容吗？”触发了闲聊路径
print_state_summary(
    run_graph_debug('session2', '知识库里有什么关于Python的内容？'),
    '测试2: 工具调用路径'
)

'''终端输出：
[root@iZbp12zrjuiewhsqxvgm4yZ personal_assistant]# uv run 6-1test.py 
===闲聊===
回复：你好！我是你的个人 AI 助理，可以帮你解答问题、整理信息、提供建议，或者协助处理一些日常任务。有什么需要帮忙的，尽管告诉我！😊

工具调用
回复：抱歉，我目前无法直接查看知识库的内容。请问您想了解什么具体信息？我可以帮您检索相关内容。

[root@iZbp12zrjuiewhsqxvgm4yZ personal_assistant]# uv run 6-1test.py 

============================================================
  测试1: 闲聊路径
============================================================
  意图 (intent)      : general_chat
  目标工具 (target)  : (无)
  工具结果数         : 0
  历史消息数         : 2
  记忆上下文长度     : 0 字符
  循环次数 (turn)    : (未设置)
  需要继续 (continue): (未设置)
------------------------------------------------------------
  最终回复 (64 字符):
  你好！我是你的个人 AI 助理，可以帮你解答问题、整理信息、提供建议，或者协助处理一些日常任务。有什么需要帮忙的，尽管告诉我！😊
============================================================
[RAG] 正在初始化 RAG 服务...
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 0.851 seconds.
Prefix dict has been built successfully.
[RAG] BM25 索引已从 Milvus 构建完成，包含 2 个文档
[RAG] RAG 服务初始化完成

============================================================
  测试2: 工具调用路径
============================================================
  意图 (intent)      : use_tool
  目标工具 (target)  : rag_search
  工具结果数         : 1
  历史消息数         : 2
  记忆上下文长度     : 0 字符
  循环次数 (turn)    : (未设置)
  需要继续 (continue): False
------------------------------------------------------------
  最终回复 (249 字符):
  根据知识库的内容，关于 Python 的信息包括：

- **语言性质**：Python 是一门解释型、面向对象的高级编程语言。
- **首次发布**：由 Guido van Rossum 于 1991 年首次发布。
- **设计哲学**：强调代码的可读性，使用缩进来定义代码块。
- **版本更新**：Python 3.12 是 2023 年发布的重大版本，引入了更友好的错误提示、类型参数语法等新特性。

如果您想了解更具体的内容，比如 Python 的语法、应用场景或学习资源，欢迎继续提问！
============================================================
[root@iZbp12zrjuiewhsqxvgm4yZ personal_assistant]# 
'''
