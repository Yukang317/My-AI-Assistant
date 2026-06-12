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
        intent: 意图分类，general_chat（普通闲聊）/ use_tool（调用工具）/ unclear（意图不清）
        target_tool: 目标工具名，对应 TOOLS 注册表的 key。仅 intent 为 use_tool 时非空
        reason: LLM 给出的路由理由，用于调试和日志
    """
    intent: str
    target_tool: Optional[str] = None
    reason: str = ""



# ── System Prompt ─────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT = """你是一个意图路由器，负责分析用户输入并决定是否需要调用工具。

## 可用工具
{available_tools}

## 路由规则
1. **use_tool**：用户的问题需要查找资料、搜索信息、检索文档才能回答 → 选择最合适的工具，返回工具名
2. **general_chat**：用户只是在闲聊、打招呼、问简单的常识性问题 → 不需要工具，直接回复
3. **unclear**：用户意图模糊，无法判断要不要用工具 → 返回 unclear，让系统反问用户

## 输出格式
严格按 JSON 格式输出，不要输出其他内容：
{{"intent": "use_tool|general_chat|unclear", "target_tool": "工具名或null", "reason": "简短理由"}}
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






def get_route_key(target_tool: Optional[str]) -> str:
    """把 target_tool 转换成 LangGraph 条件边的路由键。用于主图构建时的条件边参数
    
    target_tool 非空 → "use_tool"（走工具执行分支）
    target_tool 为空 → "general_chat"（走直接回复分支）
    """
    return "use_tool" if target_tool else "general_chat"
