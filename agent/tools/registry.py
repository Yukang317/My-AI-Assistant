"""
工具注册表 — 名称 → 函数 的映射

新增工具 = 写一个函数 + 在这里注册一行（+ 可选的 TOOL.md 说明书）。
惰性加载：只有被调用时，才把工具的 TOOL.md 描述注入 prompt。
"""

from typing import Callable, Optional
from agent.tools.base import ToolContext, ToolResult

from agent.tools.rag_search import rag_search
from agent.tools.web_search import exa_search, tavily_search

# 工具函数类型：接收 ToolContext + 输入字符串，返回 ToolResult
ToolFunc = Callable[[ToolContext, str], ToolResult]

# BOUND 安全分类常量 —— 注册表和 bound.py 共用同一套字符串
# main_graph.tool_execute 会根据分类映射到不同的 action 描述
BOUND_READ_ONLY = "READ_ONLY"   # 只读内部数据（Milvus/PG），不碰外网
BOUND_NETWORK = "NETWORK"       # 需要发 HTTP 请求的外部搜索
BOUND_WRITE = "WRITE"           # 会写文件/数据库（预留，阶段 6.4+ 用）



# ── 工具注册表 ──────────────────────────────────────────────
# key = 工具名（LLM 路由到这个名称）
# value = {"func": 函数, "description": 简短描述, "bound": BOUND分类, "skill_md": 可选TOOL.md路径}
TOOLS: dict[str, dict] = {
    "rag_search": {
        "func": rag_search,
        "description": "基于 RAG 知识库搜索文档内容，适合回答需要查找资料的问题",
        "bound": "READ_ONLY",       # 要和  agent/bound.py 里的分类一致
        "skill_md": None,
    },
    "exa_search": {
        "func": exa_search,
        "description": (
            "Exa 神经语义搜索：用 Embedding 做语义匹配，擅长找到「概念相关但用词不同」的内容。"
            "适合跨领域类比、学术检索、发散思考。需要 EXA_API_KEY"
        ),
        # ★ H1 改动：外网搜索工具标记为 NETWORK
        # 这样 tool_execute 会用「网络搜索」作为 action，而不是 NEVER_DO 里的「发送HTTP请求」
        "bound": BOUND_NETWORK,
        "skill_md": None,
    },
    "tavily_search": {
        "func": tavily_search,
        "description": (
            "Tavily AI 搜索：专为 AI Agent 设计的实时搜索，返回结构化结果（含相关性评分）。"
            "适合获取最新事实、新闻、实时数据、事实核查。需要 TAVILY_API_KEY"
        ),
        # ★ H1 改动：同上，Tavily 也是外网 HTTP 调用
        "bound": BOUND_NETWORK,
        "skill_md": None,
    },
    "document_search": {
        "func": None,
        "description": "文档内容检索，在已上传的文档中搜索相关段落（阶段 6.2 实现）",
        "bound": BOUND_READ_ONLY,
        "skill_md": None,
    },
}


def register_tool(name: str, func: ToolFunc, description: str, bound: str, skill_md: Optional[str] = None) -> None:
    """向工具注册表添加一个新工具

    Args:
        name: str,           # 工具名，也作为 TOOLS 字典的 key
        func: ToolFunc,      # 工具函数，接收 ToolContext+str，返回 ToolResult
        description: str,    # 简短描述，LLM 选工具时读
        bound: str,          # BOUND 分类，（如 "READ_ONLY", "WRITE", "NETWORK"）
        skill_md: Optional[str] = None,  # 可选的 TOOL.md 说明书路径
    
    Note:
        如果 name 已存在，打印警告后覆盖旧工具
    """
    if name in TOOLS:
        print(f"工具 '{name}' 已存在，将被覆盖。")
    TOOLS[name] = {
        "func": func,
        "description": description,
        "bound": bound,
        "skill_md": skill_md,
    }


def get_tool(name: str) -> ToolFunc:
    """根据名称获取工具函数。

    Args:
        name: 工具名称

    Returns:
        可调用的工具函数 (ToolContext, str) -> ToolResult

    Raises:
        KeyError: 工具不存在时，附带所有可用工具名列表
    """
    if name not in TOOLS:
        available = list(TOOLS.keys())
        raise KeyError(f"工具 '{name}' 不存在。可用工具：{available}")
    return TOOLS[name]["func"]


def get_tool_bound(name: str) -> str:
    """根据工具名获取 BOUND 安全分类。

    Harness 用途：tool_execute 在执行工具前先查分类，
    再决定用什么 action 描述去调 check_bound()。

    Args:
        name: 工具名称（TOOLS 字典的 key）

    Returns:
        BOUND 分类字符串，如 "READ_ONLY"、"NETWORK"

    Raises:
        KeyError: 工具不存在时，附带所有可用工具名列表
    """
    # 和 get_tool() 同一套错误处理模式，方便调试时一眼看出是「工具名写错了」
    if name not in TOOLS:
        available = list(TOOLS.keys())
        raise KeyError(f"工具 '{name}' 不存在。可用工具：{available}")
    # 只返回 bound 字段，不返回 func —— 执行前安检不需要函数本身
    return TOOLS[name]["bound"]



def list_tools() -> list[dict]:
    """列出所有已实现的工具。

    Return:
        工具信息列表，每项包含 name 和 description。
        只返回 func 不为 None 的工具（已实现），过滤占位工具。
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in TOOLS.items()
        if info["func"] is not None         # 明确可用的函数
    ]
