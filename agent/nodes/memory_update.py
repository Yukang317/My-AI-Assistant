"""
记忆更新节点 — 对话结束后增量维护 MEMORY.md

在 Graph 中作为最后一个节点，位于 result_synthesis 之后：
  判断本轮是否有值得记录的新信息 → 抽取结构化补丁 → 合并写回 MEMORY.md

半自动机制（阶段 6.2 起步）：先写入 pending 补丁到 State，由前端/API 确认后再落盘。
全自动写回在验证稳定后开启。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import MainState, StateField

# ── 常量 ──────────────────────────────────────────────────────

MEMORY_PATH: str = "/root/assistant/personal_assistant/memory/MEMORY.md"
MAX_MEMORY_CHARS: int = 2000  # ~500 token 量级，中文约 4 字/token
AUTO_APPLY_MEMORY: bool = True  # 阶段 6.2 全自动模式，后续可改为 False

# 记忆抽取器的提示词
EXTRACT_SYSTEM_PROMPT = """你是记忆抽取器，只负责从对话中提取值得长期记住的用户信息。
## 已有记忆（不要重复输出其中已有内容）
{current_memory}
## 输出规则
1. 只输出 JSON 数组，不要任何解释文字
2. 格式：[{{"section": "段落名", "content": "一条 bullet 内容"}}]
3. section 只能是以下之一：当前焦点、沟通偏好、状态/背景、待探索
4. 本轮对话没有值得记录的新信息时，输出空数组：[]
5. 不要记录一次性闲聊（如"你好"），要记录稳定的用户事实、偏好、目标、状态
## 什么值得记（示例）
- 用户在做 XX 项目，目标赶 XX 秋招 → section: 当前焦点
- 用户讨厌说教、喜欢直给结论 → section: 沟通偏好
- 用户有抑郁焦虑，任务要小步 → section: 状态/背景
"""

EXTRACT_USER_TEMPLATE = """## 本轮对话
用户：{user_input}
助手：{assistant_reply}
请输出 JSON 数组（无新信息则 []）："""



# ── bullet 相似度去重 ─────────────────────────────────────────
# 背景：记忆抽取器是 LLM，同一个事实它每次可能换个说法输出。
# 旧逻辑只做精确匹配，换说法就拦不住，去重（bullet in section_lines），换说法就拦不住，
# 这里用「归一化 + 子串 + 短语子集 + 字符 Jaccard」做近似去重，相似的一组只保留信息最全（最长）的一条。
# 注：彻底的语义合并由阶段 7.2 的 LLM 总结提炼负责，这里只做工程层兜底。

SIMILARITY_THRESHOLD: float = 0.75  # Jaccard 相似度：字符集合A、B，Jaccard(A, B) = |A ∩ B| / |A ∪ B| = 交集大小 / 并集大小。


# re.compile(pattern) 把一段正则表达式预编译成一个"正则对象"，后续_PHRASE_SEP.split(text)不用每次都编译。

# 用于切分**短语**的中英文分隔符（顿号、逗号、分号、斜杠、空白等）—— 按照标点符号切分句子为短语字符串
_PHRASE_SEP = re.compile(r"[、，,;；/。！!？?\s]+")
# 归一化时要剔除的标点（只保留用于比较的实义字符）—— 只保留文字
_PUNCT = re.compile(r"[\s，。、,.!?；;：:（）()【】\[\]\"'`*\-]+")


def _normalize_for_dedup(text: str) -> str:
    """把一条 bullet 归一化成"只剩实义字符"(只保留文字)的串，用于相似度比较。"""
    t = text.strip()
    if t.startswith("- "):
        t = t[2:]   # 砍掉markdown的“- ”前缀
    return _PUNCT.sub("", t).lower()    # 剔除标点


def _phrases(text: str) -> set[str]:
    """把一条 bullet 拆成短语集合，用于"短语子集"判断。"""
    t = text.strip()
    if t.startswith("- "):
        t = t[2:]   # 砍掉markdown的“- ”前缀
    return {p for p in _PHRASE_SEP.split(t) if p.strip()}    # 按照标点符号切分句子为短语字符串，拼成集合


def _bullet_similarity(a: str, b: str) -> float:
    """估算两条 bullet 的相似度，返回 0~1。

    判定顺序（命中即返回 1.0，视为重复）：
      1. 归一化后完全相等 —— LLM 原样重复
      2. 一条是另一条的子串 —— 扩写/缩写
      3. 较短一条的短语集合是较长一条的子集 —— 信息被完全包含
    都不命中时退回字符级 Jaccard 相似度。
    """
    # 1. 归一化后完全相等 —— LLM 原样重复
    na, nb = _normalize_for_dedup(a), _normalize_for_dedup(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    # 2. 一条是另一条的子串 —— 扩写/缩写
    if na in nb or nb in na:
        return 1.0

    # 3. 较短一条的短语集合是较长一条的子集 —— 信息被完全包含
    pa, pb = _phrases(a), _phrases(b)
    if pa and pb and (pa <= pb or pb <= pa):
        return 1.0

    # 4. 字符级 Jaccard 相似度
    sa, sb = set(na), set(nb)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _dedup_bullets(bullets: list[str], threshold: float = SIMILARITY_THRESHOLD) -> list[str]:
    """对一组 bullet 做近似去重：相似的一组只保留最长（信息最全）的一条。

    遍历每条 bullet → 跟 kept 里已保留的逐一比对 → 相似则择优替换，不相似则追加。 O(n²)复杂度在2000字符内毫秒级忽略不计。

    Args:
        bullets: 待去重的 bullet 列表
        threshold: 相似度阈值，默认 SIMILARITY_THRESHOLD

    Returns:
        去重后的 bullet 列表
    """
    kept: list[str] = []
    for bullet in bullets:
        matched = False
        for i, existing in enumerate(kept):     # 跳过第一条
            if _bullet_similarity(bullet, existing) >= threshold:
                if len(bullet.strip()) > len(existing.strip()): # 保留完整版
                    kept[i] = bullet  # 新条目信息更全，替换
                matched = True
                break
        if not matched:
            kept.append(bullet)
    return kept


# ── 补丁数据结构 ──────────────────────────────────────────────

def parse_patch_json(raw: str) -> list[dict[str, str]]:
    """解析 LLM 返回的记忆补丁 JSON。

    期望格式：[{"section": "当前焦点", "content": "..."}, ...]
    解析失败或模型判断"无需更新"时返回空列表。

    Args:
        raw: LLM 原始输出文本（可能含 markdown 代码块包裹）

    Returns:
        结构化补丁列表，每项含 section 和 content 字段
    """
    # TODO(human): 清洗 raw（去 ```json 包裹）→ json.loads → 校验结构 → 返回列表或 []
    text = raw.strip()

    # 剥掉 ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.split("\n")

        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        patches = json.loads(text)
    except json.JSONDecodeError:
        return []       # 解析失败则安全返回空列表

    # 约定好 LLM 必须输出JSON 数组 [{}, {}]
    if not isinstance(patches, list):
        return []
    
    # 校验每项结构，过滤非法条目
    valid_patches = []
    for patch in patches:
        if (
            isinstance(patch, dict) and
            "section" in patch and
            "content" in patch and
            isinstance(patch["section"], str) and
            isinstance(patch["content"], str) and
            patch["content"].strip()        # 丢弃全为空白内容的content
        ):
            valid_patches.append({"section": patch["section"].strip(), "content": patch["content"].strip()})
    
    return valid_patches


def extract_memory_patch(
    user_input: str,
    assistant_reply: str,
    current_memory: str,
    model: BaseChatModel,
) -> list[dict[str, str]]:
    """调用 LLM，从本轮对话中抽取值得长期记住的增量信息。

    对比已有 MEMORY.md 内容，只输出"新增或需要更新的条目"，
    不重复已有信息。无价值信息时返回空列表。

    Args:
        user_input: 用户本轮提问原文
        assistant_reply: AI 本轮回复原文
        current_memory: 当前 MEMORY.md 全文
        model: LangChain ChatModel 实例（建议用 DeepSeek，省 token）

    Returns:
        补丁列表，格式同 parse_patch_json 返回值
    """
    # TODO(human): 构造抽取 prompt → model.invoke → 调用 parse_patch_json 解析结果
    memory_context = current_memory.strip() or "（暂无已有记忆）"

    extract_prompt = EXTRACT_SYSTEM_PROMPT.format(current_memory=memory_context)
    user_prompt = EXTRACT_USER_TEMPLATE.format(user_input=user_input, assistant_reply=assistant_reply)
    
    messages = [SystemMessage(content=extract_prompt), HumanMessage(content=user_prompt)]
    response = model.invoke(messages)

    # 确保 response.content 是字符串，避免非字符串类型导致 json.loads 失败
    raw_patches = response.content if isinstance(response.content, str) else str(response.content)
    return parse_patch_json(raw_patches)


def merge_patch_into_memory(current_memory: str, patches: list[dict[str, str]]) -> str:
    """将补丁合并进 MEMORY.md 文本。

    按 section 字段定位已有段落并追加/更新条目；
    section 不存在时在文末新建段落。合并后若超长，
    由 truncate_memory 裁剪。

    Args:
        current_memory: 当前 MEMORY.md 全文
        patches: extract_memory_patch 产出的补丁列表

    Returns:
        合并后的 MEMORY.md 全文
    """
    # TODO(human): 按 section 合并条目，保持 Markdown 结构整洁
    # 1. 如果没有新补丁，直接返回原记忆
    if not patches:
        return current_memory

    # 2. 先按**行**处理原记忆内容的 Markdown， 比直接字符串拼接更容易控制段落位置
    lines = current_memory.splitlines()

    # 3. 遍历每条补丁，取出分区名、内容并清除首尾多余空格。
    for patch in patches:
        section = patch["section"].strip()
        content = patch["content"].strip()

        # 4. 统一成 Markown bullet（无序列表），避免 LLM 偶尔返回不带 "- " 的内容，与MEMORY.md的内容保持一致
        bullet = content if content.startswith("- ") else f"- {content}"

        section_title = f"## {section}"
        section_index = None        # 用来存这个标题在 lines 列表里的行下标，初始为空

        # 5. 找到目标 section 的标题行
        for index, line in enumerate(lines):
            if line.strip() == section_title:       # 如果找到标题行，记录标题行的下标
                section_index = index
                break

        # 6.1 如果 MEMORY.md 里还没有这个 section（即某一“##”级的部分），就在文末新建
        if section_index is None:
            # 如果MEMORY.md文档末尾有内容，先追加空行做分隔
            if lines and lines[-1].strip():
                lines.append("")
            # 追加标题行和空行
            lines.append(section_title)
            lines.append("")
            # 追加 bullet （新增 patches 的 content 经过 markdown 变成的无序列表格式）
            lines.append(bullet)
            continue
        
        # 6.2 找到当前 section 的结束位置：即下一个 "## " 标题之前
        insert_index = len(lines)
        for index in range(section_index + 1, len(lines)):
            if lines[index].startswith("## "):
                insert_index = index
                break   
        # # 7. 截取当前分区所有行，避免重复写入完全相同的 bullet
        # section_lines = lines[section_index:insert_index]       # 即截取当前分区完整范围，标题加内容。
        # if bullet in section_lines:        # 如果新增的 bullet 已经存在（即已添加过），则跳过，避免重复写入
        #     continue
        # 7. 截取当前分区所有行（含标题、空行、已有 bullet）
        section_lines = lines[section_index:insert_index]

        # 8. 如果 section 里还是模板占位符，就删除它
        # section_lines_without_placeholder = [
        # line for line in section_lines if line.strip() != "- （待记录）"

        # # 8. 收集分区内已有 bullet（剔除模板占位符），再把本次新 bullet 加进去
        existing_bullets = [
            line for line in section_lines[1:]
            if line.strip().startswith("- ") and line.strip() != "- （待记录）"
        ]
        existing_bullets.append(bullet)

        # 9. 相似度去重：既拦住本次新增的近义重复，也顺手清理历史遗留的近似条目
        deduped_bullets = _dedup_bullets(existing_bullets)

        # 关键代码：重组整个 lines，把新 bullet 塞进分区末尾
        # 10. 重建分区为「标题 + 空行 + 去重后的 bullet」，重组整个 lines
        new_section_block = [section_title, ""] + deduped_bullets
        lines = (
            # lines[:section_index]                   # 此分区section之前的内容
            # + section_lines_without_placeholder     # 此分区清理完占位符的原有分区内容
            # + [bullet]                              # ===============追加本次新条目（重点！已有分区在这里新增）===================
            # + lines[insert_index:]                  # 此分区section之后的内容
            lines[:section_index]   # 此分区之前的内容
            + new_section_block     # 去重重建后的本分区
            + lines[insert_index:]  # 此分区之后的内容
        )
    
    # 9. 将所有行拼接成字符串，并添加末尾空行
    merged = "\n".join(lines).strip() + "\n"

    # 合并后控制 MEMORY.md 长度，避免每轮对话都把上下文撑大
    return truncate_memory(merged)





def truncate_memory(content: str, max_chars: int = MAX_MEMORY_CHARS) -> str:
    """超限时裁剪 MEMORY.md，优先保留靠前的核心段落。

    阶段 6.2 用最简策略：截断到 max_chars 并加省略提示。
    后续可升级为按 section 优先级淘汰旧条目。

    Args:
        content: 待裁剪的 MEMORY.md 全文
        max_chars: 字符上限，默认 MAX_MEMORY_CHARS

    Returns:
        裁剪后的文本
    """
    # TODO(human): 未超限原样返回；超限则截断并追加省略说明
    # 未超限：原样返回（注意末尾换行一致）
    text = content.strip()
    if len(text) <= max_chars:
        return text + "\n"

    # 超限：截断，并留一点空间给省略提示
    suffix = "\n\n...（记忆已截断，较早内容已省略）"
    # 保证「正文 + suffix」不超过 max_chars
    cut_len = max_chars - len(suffix)
    if cut_len < 0:
        cut_len = 0

    truncated = text[:cut_len].rstrip() + suffix
    return truncated + "\n"
    


def write_memory_file(content: str, path: str = MEMORY_PATH) -> None:
    """将内容写入 MEMORY.md，目录不存在时自动创建。

    Args:
        content: 要写入的 MEMORY.md 全文
        path: 目标文件路径，默认 MEMORY_PATH

    Raises:
        OSError: 文件写入失败时抛出
    """
    # TODO(human): os.makedirs 确保目录存在 → 写入文件
    # os.path.dirname 取出目录部分，例如 .../memory/MEMORY.md → .../memory
    directory = os.path.dirname(path)

    # exist_ok=True：目录已存在也不报错
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    # encoding="utf-8"：确保写入文件时使用 UTF-8 编码，避免中文乱码
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def memory_update(state: MainState, model: BaseChatModel) -> dict:
    """LangGraph 记忆更新节点：抽取补丁 → 合并 → 写回 MEMORY.md。

    读取 State 中的 user_question、final_response、memory_context，
    调用 extract_memory_patch 判断是否有增量，有则合并写回。

    阶段 6.2 起步策略：auto_apply=True 时直接写盘；
    否则将 pending_patch 写入 State 等待确认（半自动）。

    Args:
        state: 当前 MainState，需含 user_question / final_response / memory_context
        model: LangChain ChatModel 实例

    Returns:
        dict: 含 MEMORY_CONTEXT（更新后）和可选的 MEMORY_PENDING_PATCH 字段
    """
    # TODO(human): 取 state 数据 → extract → merge → write → 返回更新后的 memory_context
    # 1. 从 State 取本轮数据（load_context / result_synthesis 已写入）
    user_input = state.get(StateField.USER_QUESTION, "")
    assistant_reply = state.get(StateField.FINAL_RESPONSE, "")
    memory_context = state.get(StateField.MEMORY_CONTEXT, "")

    # 2. 缺关键字段就跳过，避免空对话也调 LLM 浪费 token
    if not user_input.strip() or not assistant_reply.strip():
        return {StateField.MEMORY_CONTEXT: memory_context}
    
    # 3. 调 LLM 抽取增量补丁
    patches = extract_memory_patch(
        user_input=user_input,
        assistant_reply=assistant_reply,
        current_memory=memory_context,
        model=model,
    )

    # 4. 如果没新补丁，直接返回原记忆
    if not patches:
        return {StateField.MEMORY_CONTEXT: memory_context}

    # 5. 半自动模式：暂不写盘，只把补丁留给后续确认 API （阶段 6.2 后期再接）
    if not AUTO_APPLY_MEMORY:
        return{
            StateField.MEMORY_CONTEXT: memory_context,
            "memory_pending_patches": patches,      # # 后续可加到 StateField 枚举
        }

    # 6. 全自动模式：合并 -> 写盘，返回更新后的记忆
    merged = merge_patch_into_memory(memory_context, patches)
    write_memory_file(content=merged)
    return {StateField.MEMORY_CONTEXT: merged}