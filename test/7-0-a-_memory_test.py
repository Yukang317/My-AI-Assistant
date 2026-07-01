from __future__ import annotations

import re

# ═══════════════════════════════════════════════════════════════════════
# §一 常量 + 四个辅助函数 — 复制到 memory_update.py
# ═══════════════════════════════════════════════════════════════════════

# 背景：记忆抽取器是 LLM，同一个事实它每次可能换个说法输出。
# 旧逻辑只做 bullet in section_lines 精确匹配，换说法就拦不住。
# 这里用「归一化 + 子串 + 短语子集 + 字符 Jaccard」做工程层近似去重。
# 彻底的语义合并由阶段 7.2 的 LLM 总结提炼负责。

SIMILARITY_THRESHOLD: float = 0.75  # Jaccard ≥ 此值视为重复

# 切分短语用的分隔符（顿号、逗号、分号、斜杠、空白等）
_PHRASE_SEP = re.compile(r"[、，,;；/\s]+")
# 归一化时剔除的标点，只留实义字符做比较
_PUNCT = re.compile(r"[\s，。、,.!?；;：:（）()【】\[\]\"'`*\-]+")


def _normalize_for_dedup(text: str) -> str:
    """把一条 bullet 归一化成「只剩实义字符」的串，用于相似度比较。

    例："- 喜欢小步教学、讨厌说教" → "喜欢小步教学讨厌说教"
    """
    t = text.strip()
    if t.startswith("- "):
        t = t[2:]  # 去掉 Markdown bullet 前缀，只比内容
    return _PUNCT.sub("", t).lower()


def _phrases(text: str) -> set[str]:
    """把 bullet 拆成短语集合，用于「短语子集」判断。

    例："喜欢小步教学、讨厌说教" → {"喜欢小步教学", "讨厌说教"}
    """
    t = text.strip()
    if t.startswith("- "):
        t = t[2:]
    return {p for p in _PHRASE_SEP.split(t) if p.strip()}


def _bullet_similarity(a: str, b: str) -> float:
    """估算两条 bullet 的相似度，返回 0.0~1.0。

    判定顺序（命中即 1.0，视为重复）：
      1. 归一化后完全相等 — LLM 原样重复
      2. 一条是另一条的子串 — 扩写/缩写（"喜欢小步教学" ⊂ "用户叫 Yukang，喜欢小步教学"）
      3. 较短条的短语集合是较长条的子集 — 信息被完全包含
    都不命中时退回字符级 Jaccard（set 交集/并集）。
    """
    na, nb = _normalize_for_dedup(a), _normalize_for_dedup(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 1.0

    pa, pb = _phrases(a), _phrases(b)
    if pa and pb and (pa <= pb or pb <= pa):
        return 1.0

    sa, sb = set(na), set(nb)
    union = len(sa | sb)
    return len(sa & sb) / union if union else 0.0


def _dedup_bullets(bullets: list[str], threshold: float = SIMILARITY_THRESHOLD) -> list[str]:
    """对一组 bullet 近似去重：相似的一组只保留最长（信息最全）的一条。

    保持首次出现顺序。遇到与已保留条目相似的新条目时：
      - 新条目更长 → 替换旧条目
      - 否则 → 丢弃新条目
    """
    kept: list[str] = []
    for bullet in bullets:
        matched = False
        for i, existing in enumerate(kept):
            if _bullet_similarity(bullet, existing) >= threshold:
                if len(bullet.strip()) > len(existing.strip()):
                    kept[i] = bullet  # 新条目信息更全，替换
                matched = True
                break
        if not matched:
            kept.append(bullet)
    return kept


def _self_test() -> None:
    """5 个场景验证去重逻辑，全部 pass 再手敲正式代码。"""
    cases = [
        # (输入 bullets, 期望保留条数, 描述)
        (
            [
                "- 喜欢小步教学、讨厌说教",
                "- 用户叫 Yukang，喜欢小步教学、讨厌说教",
            ],
            1,
            "子串/包含关系 → 保留更长那条",
        ),
        (
            [
                "- 讨厌说教",
                "- 喜欢小步教学、讨厌说教",
            ],
            1,
            "短语子集 → 保留更长那条",
        ),
        (
            ["- 在做 personal_assistant 项目", "- 在做 personal_assistant 项目"],
            1,
            "完全相同 → 只留 1 条",
        ),
        (
            ["- 目标赶秋招", "- 讨厌说教"],
            2,
            "完全不同 → 都保留",
        ),
    ]

    for bullets, expected_count, desc in cases:
        result = _dedup_bullets(bullets)
        assert len(result) == expected_count, (
            f"FAIL [{desc}]: 期望 {expected_count} 条，实际 {len(result)} 条 → {result}"
        )
        print(f"  ✅ {desc} → {result}")

    # 真实 MEMORY.md 三条目：阈值 0.75 下 3→2（最长那条与中间条 Jaccard 仅 ~0.47）
    # 彻底合并靠 7.2 LLM 总结；7.0 工程层已拦住最明显的重复
    real = [
        "- 喜欢小步教学、不说废话、直接上手，讨厌说教和啰嗦",
        "- 喜欢小步教学、讨厌说教",
        "- 用户叫 Yukang，喜欢小步教学、讨厌说教",
    ]
    real_result = _dedup_bullets(real)
    assert len(real_result) == 2, f"MEMORY.md 场景: 期望 2 条，实际 {len(real_result)}"
    print(f"  ✅ MEMORY.md 三条目 → {len(real_result)} 条 → {real_result}")

    print("\n全部 5 场景通过。")


if __name__ == "__main__":
    print("7-0-a 去重逻辑自测：")
    _self_test()
