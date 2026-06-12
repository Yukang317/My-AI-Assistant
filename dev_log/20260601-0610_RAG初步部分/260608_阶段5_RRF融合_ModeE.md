# 阶段5 RRF 融合排序 — Mode E 编码完成

**日期**：2026-06-08
**文件**：`rag/retrieval/rrf_fusion.py`
**上下文**：Mode E 骨架式教学，路线2（知识检索）第三批文件

---

## 架构角色

```
路线2（知识检索）检索管线：
                              ┌── bm25.py ──────────→ BM25 关键词排名 ──┐
用户查询 ──→                  │                                           ├──→ rrf_fusion.py ──→ 重排
                              └── vector_retriever.py → 向量语义排名 ──┘
```

RRF 不关心两个通道的原始分数大小，只关心排名——天然解决异质分数融合问题。

---

## RRFFusion — 4 个函数全部完成

| 函数 | 说明 |
|------|------|
| `__init__` | 保存 k 值（默认60），加 k>0 校验 |
| `_build_rank_map` ★关键 | chunk 排名列表 → 文档排名表，按 doc_id 去重保留最佳排名 |
| `fuse` ★核心 | 两通道 rank map → RRF 公式 → 排序 → top_k |
| `get_stats` | 返回 k 值和算法名 |

### 核心数据流（fuse 调用链）

```
bm25_results                     vector_results
  [{chunk_index, doc_id,           [{chunk_id, doc_id,
    score, text}, ...]               score, text, doc_name, doc_type}, ...]
        │                                  │
        ▼                                  ▼
_build_rank_map(..., "doc_id")    _build_rank_map(..., "doc_id")
        │                                  │
        ▼                                  ▼
{"deepseek.md": {rank:1, ...},    {"python.md": {rank:1, ...},
 "python.md":   {rank:2, ...}}     "deepseek.md": {rank:3, ...},
                                   "docker.md":   {rank:2, ...}}
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
         all_doc_ids = {"deepseek.md", "python.md", "docker.md"}
                   │
                   ▼
         对每个 doc_id 算 RRF_score:
           deepseek.md: 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
           python.md:   1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325
           docker.md:   0          + 1/(60+2) = 0          + 0.0161 = 0.0161
                   │
                   ▼
         排序: python.md(0.0325) > deepseek.md(0.0323) > docker.md(0.0161)
                   │
                   ▼
         return 前 top_k 条
```

---

## 已覆盖知识点

| 函数 | 关键知识点 |
|------|-----------|
| `__init__` | k=60 来自 SIGIR 2009 论文实验结论；k 是"阻尼器"，越大越民主；k=0 会让第一名垄断 |
| `_build_rank_map` | enumerate start=1 做 1-based rank；`item.get(field)` 防御缺失字段；`**item` 字典解包注入 rank；O(1) 字典 in 做去重；"约定优于配置"——信任输入已排序 |
| `fuse` | RRF 公式 `Σ 1/(k+rank)`；`set() \| set()` 取并集；安全取值：`.get()` + 海象运算符 `:=`；text 优先取 vector（语义 chunk 更完整）；doc_name/doc_type 只在 vector 有；循环内初始化累加变量 |
| `get_stats` | 配合 BM25/Vector 的 get_stats 形成检索链可观测性 |

### RRF 为什么优于分数归一化

| 方法 | 问题 |
|------|------|
| Min-Max 归一化 | 受异常值影响大（一个极端高分把所有其他分数压到接近0） |
| Softmax | 对分数分布形状敏感（偏态分布结果不稳定） |
| **RRF（排名法）** | 分布无关；异常值不影响排名；跨系统天然可比 |

---

## 融合粒度决策

采用**文档级融合**（以 doc_id 为键）：

| 方案 | 键 | 问题 |
|------|-----|------|
| 文档级（选定） | `doc_id` | 一个文档多个 chunk 只保留最佳排名 |
| chunk 级（不可行） | `chunk_id` | BM25 用 chunk_index（整数），Vector 用 chunk_id（字符串），无法匹配 |

---

## Bug 复盘

### Bug 1: `rrf_score` 累加不重置（fuse 函数）

**现象**：每个 doc_id 的 RRF 分数等于前面所有文档分数的累加和
**根因**：`rrf_score = 0.0` 写在 for 循环外面
**修复**：移到循环体第一行

```python
# ❌ 错误
rrf_score = 0.0
for doc_id in all_doc_ids:
    rrf_score += 1.0 / (self.k + ...)  # 每次都在累加！

# ✅ 正确
for doc_id in all_doc_ids:
    rrf_score = 0.0                     # 每个文档独立
    rrf_score += 1.0 / (self.k + ...)
```

### Bug 2: 排序缺失

**现象**：`items[:top_k]` 返回乱序结果
**根因**：排序代码被漏掉，后面是两行死代码
**修复**：`items.sort(key=lambda x: x["rrf_score"], reverse=True)`

### Bug 3: doc_name/doc_type KeyError

**现象**：文档只在 BM25 结果中时，`bm25_rank_map[doc_id]["doc_name"]` → KeyError
**根因**：BM25 结果没有 doc_name/doc_type 字段，fallback 取 BM25 的值报错
**修复**：`vector_info.get("doc_name") if vector_info else None`

### Bug 4: _build_rank_map 缩进错误（用户自主修复前）

**现象**：rank_map 永远为空
**根因**：`rank_map[item_id] = {...}` 缩进在 `if item_id in rank_map: continue` 块内，成为死代码
**修复**：缩进与 if 对齐

---

## 协作模式记录

- 用户对 `_build_rank_map` 的作用不理解，用具体数据例子推演后理解
- 用户要求"前因后果一句话总结"，产出了函数注释的紧凑描述
- `fuse` 函数写到一半卡住，Claude 用具体数值推演（deepseek.md/python.md/docker.md 三个文档的 RRF 计算）帮助突破
- 3 个 bug 由 Claude 审查发现并修复（rrf_score 累加、排序缺失、doc_name KeyError）

---

## 下一步

Step 8（检索通道）全部完成：bm25.py ✅ + vector_retriever.py ✅ + rrf_fusion.py ✅

下一步：**Step 9** `rag/rag_service.py` — RAG 服务编排层，串联整个管线。

管线全貌：
```
查询 → BM25 + Vector 并行 → RRF 融合 → (重排) → 父块回溯 → LLM 生成
 ⬜    ✅    ✅             ✅          ⬜        ⬜         ⬜
```
