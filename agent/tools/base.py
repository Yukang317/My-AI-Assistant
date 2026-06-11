"""
工具接口定义 — ToolResult 数据类 + 工具函数签名约定

设计原则：
- 每个工具函数接收 ToolContext + 输入字符串，返回 ToolResult
- ToolResult 统一格式：success/error/data/artifacts
- 不做异步（阶段 6.1 全部同步，与现有 DB/RAG 模块保持一致）
"""

from dataclasses import dataclass, field
from typing import Any, Optional

# 项目里工具调用的数据需要跨越多个 LangGraph 节点传递，dataclass 自带的 __init__、__repr__、__eq__ 让调试和日志输出更清晰
# 例如：执行完工具后 LangGraph 打印 state 时：
#       tool_results=[ToolResult(success=True, data='RTX 5090 的价格是...', artifacts={'sources': [...]})]
@dataclass
class ToolContext:
    """传递给每个工具调用的上下文信息。

    包含当前会话 ID 和用户 ID，工具可以据此访问数据库、读取用户偏好等。
    阶段 6.1 只用到 session_id，后续扩展 user_id。
    """
    session_id: str
    user_id: Optional[str] = None


@dataclass
class ToolResult:
    """工具执行结果统一格式。

    Attributes:
        success: 是否成功执行
        data: 返回给 LLM 的主要数据（字符串或字典）
        error: 失败时的错误信息
        artifacts: 附带产物（文件路径、URL 等），可选
    """
    success: bool                                   
    data: Any
    error: Optional[str] = None
    artifacts: dict = field(default_factory=dict) # 每次实例化才会调 dict() 创建新对象，互不干扰。


# 工具函数类型签名约定
# 所有工具函数遵循: (ToolContext, str) -> ToolResult

'''
success、data：无默认值，每个新的实例工具都会创建新的内存空间存储自己的相关值
error：默认 None，python 中 None 指向同一个内存空间不可被修改
artifacts： 有默认值为空，字典可被修改，所以每次实例都会指向并修改同一内存空间的值，导致污染
另外：
如果某个实例执行了赋值操作：
r1.artifacts = {"new": "dict"}   # 注意，这是赋值，不是修改原字典
那么 r1.artifacts 会指向一个全新的字典，不再与其他实例共享。但这通常不是我们想要的，我们想要的是每个实例默认拥有独立的空字典，这正是 field(default_factory=dict) 解决的问题。
'''



