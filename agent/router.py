"""
意图路由 — LLM 分析用户输入，判断意图并选择对应工具

借鉴 sagt_agent 的 intent_detection 节点设计：
- 用 LangChain ChatModel 做意图识别
- 返回目标工具名 + 路由键
- 不强制依赖路由准确性（有 fallback 兜底）
"""

from typing import Optional
from langchain_core.language_models import BaseChatModel

from dataclasses import dataclass
import json 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ── 路由结果 ──────────────────────────────────────────────────

@dataclass
class RouteResult:
    """LLM 意图路由结果。

    Attributes:
        intent: 意图分类——
            general_chat（普通闲聊，不需要工具）
            use_tool（事实性查询，需要搜索资料回答）
            inspire（灵感/扩展思考，需要跨领域搜索 + 发散合成）
            unclear（意图不清，需要反问用户）
        target_tool: 目标工具名，对应 TOOLS 注册表的 key。
            intent=use_tool 时 LLM 选择最合适的单个工具
            intent=inspire 时 LLM 选择最合适的工具（通常 exa_search，擅长跨领域）
        reason: LLM 给出的路由理由，用于调试和日志
    """
    intent: str
    target_tool: Optional[str] = None
    reason: str = ""



# ── System Prompt ─────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """你是一个意图路由器，负责分析用户输入并决定如何处理。

## 可用工具
{available_tools}

## 路由规则（四选一）

1. **general_chat**：用户在闲聊、打招呼、问简单的常识性问题、或进行日常对话
   → 不需要工具，系统直接回复

2. **use_tool**：用户问的是事实性问题，需要查找资料、搜索信息、检索文档才能准确回答
   → 选择一个最合适的工具，比如：
   - 「RTX 5090 什么价格？」→ tavily_search（实时价格信息）
   - 「知识库里有关于微服务的文档吗？」→ rag_search（检索个人文档）
   - 「Transformer 架构的原理是什么？」→ exa_search（学术/技术概念）

3. **inspire**：用户想要发散思维、跨界思考、获取灵感或扩展视野，而非寻找单一事实答案
   → 选择一个最合适的搜索工具（通常 exa_search 更适合跨领域概念关联）
   → 例如：
   - 「给我一些关于 AI 在医疗领域的创新思路」
   - 「从生物学角度看，公司管理有什么启发？」
   - 「有什么我不知道但应该知道的关于认知偏差的研究？」
   - 「帮我扩展一下对这个问题的思考」

4. **unclear**：用户意图模糊，无法判断意图
   → 返回 unclear，让系统反问用户

## 判断优先级
- 区分 use_tool 和 inspire 的关键：用户是在**找答案**（use_tool）还是在**找思路**（inspire）？
- 「X 是什么」「X 多少钱」「X 的最新进展」→ use_tool
- 「X 有什么启发」「从 Y 角度怎么看 X」「给我一些跨领域的思路」→ inspire
- 不确定时，倾向 general_chat（让系统自然对话）而非强行分类

## 输出格式
严格按 JSON 格式输出，不要输出其他内容：
{{"intent": "general_chat|use_tool|inspire|unclear", "target_tool": "工具名或null", "reason": "简短理由"}}
"""


# ── 路由函数 ──────────────────────────────────────────────────

def route_intent(question: str, model: BaseChatModel, available_tools: list[dict]) -> RouteResult:
    """用 LLM 分析用户意图，返回路由结果。
    
    把可用工具列表注入 System Prompt，让 LLM 根据问题内容选择
    最合适的工具。JSON 解析失败时 fallback 到 general_chat。
    """
    # 1. 把工具列表拼成可读字符串
    tools_str = "\n".join(
        f"- {t['name']}:{t['description']}" for t in available_tools
    ) if available_tools else "（当前无可用工具）"

    # 2. 组装消息 + JSON prefill
    #    在消息末尾追加 AIMessage(content="{")，LLM 看到 assistant 已输出 "{"
    #    会直接续写 JSON 内容，从源头杜绝 "好的，这是JSON：" / ```json 等冗余
    system_msg = SystemMessage(
        content=ROUTER_SYSTEM_PROMPT.format(available_tools=tools_str)
    )
    messages = [
        system_msg,
        HumanMessage(content=question),
        AIMessage(content="{"),  # prefill：引导 LLM 直接输出 JSON
    ]

    # 3. 调用 LLM
    response = model.invoke(messages)
    raw = response.content.strip()

    # 4. 解析 JSON（prefill 后不再需要剥离 markdown）
    try:
        if not raw.startswith("{"):
            raw = "{" + raw  # 补回 prefill 的 "{" 前缀
        data = json.loads(raw)
        return RouteResult(
            intent=data.get("intent", "general_chat"),
            target_tool=data.get("target_tool"),
            reason=data.get("reason", ""),
        )
    except (json.JSONDecodeError, KeyError, AttributeError):
        # JSON 解析失败，安全 fallback：当普通聊天处理
        return RouteResult(
            intent="general_chat",
            reason=f"路由解析失败，fallback to general_chat。LLM 原始输出: {raw[:100]}",
        )






def get_route_key(intent: str, target_tool: Optional[str]) -> str:
    """把 intent + target_tool 转换成 LangGraph 条件边的路由键。

    路由逻辑：
    - intent="inspire"          → "inspire"         （P1 阶段走工具执行，P2/P3 升级为发散-收敛管线）
    - intent="use_tool" + 工具  → "use_tool"        （单工具调用）
    - intent="general_chat"     → "general_chat"    （直接 LLM 回复）
    - intent="unclear"          → "general_chat"    （意图不清时安全 fallback：直接回复 + 反问）
    - 其他情况                   → "general_chat"    （兜底）

    用于主图构建时 conditional_edges 的 path_map 参数。
    """
    if intent == "inspire":
        return "inspire"
    elif intent == "use_tool" and target_tool:
        return "use_tool"
    else:
        return "general_chat"
