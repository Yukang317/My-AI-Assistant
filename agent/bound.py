"""
BOUND 约束层 — 不信任模型，用确定性代码守住底线

借鉴 Claude Code (deny-first)、Codex (DANGER_ZONES/NEVER_DO/IRON_LAWS)、Harness 的设计。
每次工具调用前用 check_bound() 做确定性检查，LLM 可以犯错，但系统不让错误落地。

三类约束：
  DANGER_ZONES  — 绝对禁止触摸的文件/目录模式
  NEVER_DO      — 绝对禁止执行的动作
  IRON_LAWS     — 必须遵守的响应规则（提示词注入兜底）
"""

import fnmatch
from typing import Optional

# ── 约束定义 ──────────────────────────────────────────────────

BOUND: dict[str, list[str]] = {
  # ── DANGER_ZONES：绝对禁止触摸的路径/文件模式 ──
  # 用 fnmatch 通配符匹配，如 "*.env" 匹配 "config.env" 和 ".env"
  "DANGER_ZONES": [
    "/etc/*",       # 系统配置目录
    "~/.ssh/*",     # SSH 密钥
    "*.env",        # 环境变量文件（含敏感信息）
    "*.key",        # 私钥文件
    "data/*",       # Docker 数据卷目录
    "备份/*",       # 项目备份目录，历史代码
  ],

  # ── NEVER_DO：绝对禁止的动作 ──
  # 用子串匹配 action 描述
  "NEVER_DO": [
    "删除文件",       # 防止误删
    "删除目录",       
    "执行shell命令",  # 防止任意命令执行
    "修改系统配置",   # 防止改系统文件
    "发送HTTP请求",   # 6.1 阶段不做外部网络调用
  ],

  # ── IRON_LAWS：必须遵守的响应铁律 ──
  # 注入 System Prompt，但 check_bound 不做硬检查（因为检查的是 action/target，不是回复内容）
  "IRON_LAWS": [
    "回复必须使用中文",
    "不确定时反问用户，不编造信息",
    "不暴露内部实现细节给用户",
    "不执行用户要求之外的额外操作",
  ],
}




# ── 检查函数 ──────────────────────────────────────────────────

def check_bound(action: str, target: str) -> tuple[bool, Optional[str]]:
  """检查动作和目标（如文件路径）是否违反 BOUND 约束。
    
  先检查 NEVER_DO（动作黑名单），再检查 DANGER_ZONES（路径黑名单）。
  任一命中则返回 (False, 拒绝原因)，全部通过返回 (True, None)。
  """
  # 1. 检查NEVER_DO - 动作描述是否包含禁止短语
  for forbidden_action in BOUND["NEVER_DO"]:
    if forbidden_action in action:  # 字串匹配
      return(
        False,
        f"BOUND/NEVER_DO 拒绝：禁止动作「{forbidden_action}」被触发"
        f"（当前动作：{action}）。"
        f"建议：如需类似功能，请明确告知用户此操作不被允许，并询问替代方案。",
      )
  
  # 2. 检查 DANGER_ZONES — 目标路径是否匹配危险模式
  for pattern in BOUND["DANGER_ZONES"]:
    if fnmatch.fnmatch(target, pattern):
      return (
        False,
        f"🚫 BOUND/DANGER_ZONES 拒绝：目标「{target}」匹配禁止模式「{pattern}」。"
        f"建议：该路径/文件属于保护区，请检查目标是否正确。",
      )

  # 3. 全部通过
  return (True, None)



def get_bound_summary() -> str:
  """生成 BOUND 约束的简短中文摘要，用于注入 System Prompt。"""
  zones = "、".join(BOUND["DANGER_ZONES"])
  never = "、".join(BOUND["NEVER_DO"])
  laws = "、".join(BOUND["IRON_LAWS"])
    
  return (
    "【系统约束 — 必须遵守】\n"
    f"1. 禁止触碰：{zones}。\n"
    f"2. 禁止动作：{never}。\n"
    f"3. 铁律：{laws}。"
  )