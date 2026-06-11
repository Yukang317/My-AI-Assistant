"""
BOUND 约束层 — 不信任模型，用确定性代码守住底线

借鉴 Claude Code (deny-first)、Codex (DANGER_ZONES/NEVER_DO/IRON_LAWS)、Harness 的设计。
每次工具调用前用 check_bound() 做确定性检查，LLM 可以犯错，但系统不让错误落地。

三类约束：
  DANGER_ZONES  — 绝对禁止触摸的文件/目录模式
  NEVER_DO      — 绝对禁止执行的动作
  IRON_LAWS     — 必须遵守的响应规则（提示词注入兜底）
"""

from typing import Optional

# ── 约束定义 ──────────────────────────────────────────────────

# TODO(human): 定义 BOUND: dict[str, list[str]] 常量
# 说明：
#   1. DANGER_ZONES: 列出本项目不应该被 AI 修改的文件/目录
#      - "/etc", "~/.ssh", "*.env", "*.key"
#      - "data/" (Docker 数据目录)
#      - "备份/"
#   2. NEVER_DO: 列出绝对禁止的动作
#      - "删除文件", "执行shell命令", "修改系统配置"
#      - "发送HTTP请求到外部服务"
#   3. IRON_LAWS: 必须遵守的响应规则
#      - "回复必须使用中文"
#      - "不确定时反问用户，不编造信息"
#      - "不暴露内部实现细节给用户"


# ── 检查函数 ──────────────────────────────────────────────────

# TODO(human): 实现 check_bound(action: str, target: str) -> tuple[bool, Optional[str]]
# 说明：
#   1. 接收 action（动作描述，如 "读取文件"、"执行搜索"）和 target（目标，如文件路径）
#   2. 遍历 NEVER_DO：如果 action 匹配任何禁止动作，返回 (False, 拒绝原因)
#   3. 遍历 DANGER_ZONES：如果 target 匹配任何危险模式，返回 (False, 拒绝原因)
#      - 用 fnmatch 做通配符匹配（如 "*.env" 匹配 "config.env"）
#   4. 通过所有检查则返回 (True, None)
#   5. 返回的拒绝原因字符串要包含"哪条规则被触发"和"建议的替代方案"
#   6. 需要导入 fnmatch 模块


# TODO(human): 实现 get_bound_summary() -> str 函数
# 说明：
#   1. 生成 BOUND 约束的简短中文摘要（~200 字）
#   2. 用于注入 System Prompt，让 LLM 在生成动作前就自我约束
#   3. 摘要应列出关键禁止项和铁律，语气坚定但不生硬
