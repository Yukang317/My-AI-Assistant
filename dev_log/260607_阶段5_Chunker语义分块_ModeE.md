# 阶段5 Step 6: Chunker 语义分块 — Mode E 教学

**日期**：2026-06-07
**状态**：semantic_splitter.py ✅ 完成 | parent_child_builder.py ⬜ 待教学

---

## 做了什么

### 1. semantic_splitter.py — 3 函数全部实现

| 函数 | 核心 | 状态 |
|------|------|------|
| `split_markdown` | MarkdownHeaderTextSplitter(标题切分) → RecursiveCharacterTextSplitter(超长再切) | ✅ |
| `split_text` | RecursiveCharacterTextSplitter + 中文分隔符降级 `["\n\n","\n","。","."," ",""]` | ✅ |
| `split_by_type` | 策略分派：md/pdf/docx→markdown管道, txt/其他→text管道 | ✅ |

### 2. 架构澄清：parent_child_builder vs vector_store 父子策略分工

#### 父子策略的两个阶段

```
📦 构建阶段（chunker/parent_child_builder.py）     💾 存储+检索阶段（storage/vector_store.py）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

List[SemanticChunk]                               已构建好的 parent_chunks + child_chunks
  │  (~512 字符的语义块)                                │  (带 content_vector)
  ▼                                                  ▼
build_parent_child()                             insert_parent_child()
  │                                                  │
  ├─ 遍历语义块，拼成 ~2048 的父块                      ├─ 父块 → parent_collection
  ├─ 每个父块再切成 ~512 的子块                          ├─ 子块 → child_collection
  ├─ 生成 parent_id: "文档_parent_0"                   └─ 刷盘
  └─ 生成 child_id:  "文档_parent_0_child_2"
  │                                                  ▼
  ▼                                               search()
List[ParentChunk] + List[ChildChunk]                 │
  (纯文本，无向量)                                     ├─ query_vector → child_collection 搜
                                                     ├─ 拿到 parent_id → parent_collection 查
                                                     ├─ 同 parent_id 去重
                                                     └─ 返回 {child_content + parent_content}
```

#### 一句话总结

| | parent_child_builder.py | vector_store.py |
|---|---|---|
| **角色** | 🏗️ 建筑师 — 决定**怎么切** | 🏪 仓库管理员 — 负责**存和查** |
| **输入** | 语义块列表（纯文本） | 已构建好的块 + 向量 |
| **输出** | ParentChunk + ChildChunk 对象 | Milvus 写入 / 搜索结果 |
| **关注点** | 块大小、重叠、ID 关联 | Schema、索引、向量搜索 |

**打个比方**：parent_child_builder 是把一本书拆成"章（父块）→ 节（子块）"并贴上标签；vector_store 是把这些贴好标签的章节放进图书馆书架，然后帮读者按内容相似度找书。一个是"制造"，一个是"仓储+查询"。

---

## 知识点

### split_markdown
- **双步策略**：标题切分(保留结构) → 超长段递归切分(控制大小)。先语义后大小的两层保证
- **strip_headers=False**：RAG 场景必须保留 `## 标题` 行在 content 里，LLM 需要看到标题文字
- **dict(doc.metadata)**：LangChain 的 metadata 类型不是普通 dict，需显式转换

### split_text
- **分隔符优先级**：`\n\n → \n → 。→ . → 空格 → 字符`，从粗到细降级，优先在自然边界切
- **纯文本无标题**：metadata 只记 start_index，无 header 信息

### split_by_type
- **策略模式**：根据 doc_type 分派，pdf/docx 走 markdown 管道（pymupdf4llm/markitdown 输出已是 Markdown）
- **else 兜底**：所有未知类型走 text 降级管道，不抛异常——宽容输入

### 分层职责判断标准
- **独立变化的放不同层**：改块大小只动 chunker，换数据库只动 vector_store
- **一起变化的放同一层**：切块逻辑和 ID 生成都在 chunker

---

## 三、parent_child_builder.py（接续）

> 代码：`personal_assistant/rag/chunker/parent_child_builder.py`

### 3.1 做了什么

| 函数 | 核心 | 状态 |
|------|------|------|
| `_generate_parent_id` | f-string: `"{doc_name}_parent_{index}"` | ✅ |
| `_generate_child_id` | f-string: `"{parent_id}_child_{index}"` | ✅ |
| `build_parent_child` | 遍历语义块拼父块(2048) → 切子块(512) → ID关联 → 收尾 | ✅ |

### 3.2 核心算法流程

```
1. 异常检查 → 2. 准备 buffer/child_splitter
→ 4. _finalize_parent() 闭包（buf → 1父+N子）
   ├─ i.   生成 parent_id
   ├─ ii.  切分子文本
   ├─ iii. 创建 ChildChunk 列表
   └─ iv.  创建 ParentChunk
→ 5. 遍历语义块：超了→固化→overlap截取→拼接
→ 6. 收尾：最后残留也固化
```

### 3.3 重构讨论：该不该抽 _finalize_parent 函数？

**问题**：固化逻辑在循环和收尾各出现一次（2次），要不要提取？

**初始尝试**：用户提到文件顶层的 `_build_parent_with_children()`，但遇到问题：
- 循环里没改，重复没消除
- 返回类型注解错了
- 为适配 `.extend()` 把单对象装进列表，语义扭曲

**最终方案**：用闭包 `_finalize_parent(buf, p_idx)`，放在 `build_parent_child` 内部。返回 `(ParentChunk, List[ChildChunk])`，循环和收尾各一行调用。

**知识要点**：
- **DRY 核心不是"重复几次"，而是"是否同一个理由会变"**：固化逻辑改了，两处都得改——这是真实重复
- **闭包 vs 顶层函数**：闭包暗示"这是内部辅助件，不是公共 API"；放在文件顶层则表示"这是模块职责"
- 即使只重复 2 次，如果代表同一个概念，就该抽

### 3.4 踩坑

| # | 现象 | 根因 | 修复 |
|---|------|------|------|
| 1 | `else::` 双冒号 SyntaxError | 手误 | 改为 `else:` |
| 2 | 死代码 try/except ValueError | split_markdown/split_text 不抛 ValueError | 删掉 |
| 3 | 提取函数后循环仍内联 | 只在收尾处用，没改循环 | 闭包统一两处 |
