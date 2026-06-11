"""
工具注册表 — 名称 → 函数 的映射

新增工具 = 写一个函数 + 在这里注册一行（+ 可选的 TOOL.md 说明书）。
惰性加载：只有被调用时，才把工具的 TOOL.md 描述注入 prompt。
"""

from typing import Callable, Optional
from agent.tools.base import ToolContext, ToolResult

from agent.tolls.rag_search import rag_search

# 工具函数类型：接收 ToolContext + 输入字符串，返回 ToolResult
ToolFunc = Callable[[ToolContext, str], ToolResult]

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
    "web_search": {
        "func": None,  
        "description": "网页搜索，查询互联网上的实时信息（阶段 6.2 实现）",
        "bound": "READ_ONLY",
        "skill_md": None,
    },
    "document_search": {
        "func": None,
        "description": "文档内容检索，在已上传的文档中搜索相关段落（阶段 6.2 实现）",
        "bound": "READ_ONLY",
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


def get_tool(name: str) -> dict:
    """根据名称获取工具信息。

    Args:
        name: 工具名称
    
    Returns:
        包含 func/description/bound/skill_md 的字典

    Raises:
        KeyError: 工具不存在时，附带所有可用工具名列表
    """
    if name not in TOOLS:
        available = list(TOOLS.keys())
        raise KeyError(f"工具 '{name}' 不存在。可用工具：{available}")
    return TOOLS[name]




# TODO(human): 实现 list_tools() 函数
# 说明：
#   1. 返回 TOOLS 中所有 func 不为 None 的工具列表
#   2. 每条包含 name 和 description
#   3. 用于拼接 LLM 的 System Prompt，让 LLM 知道有哪些工具可用
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
