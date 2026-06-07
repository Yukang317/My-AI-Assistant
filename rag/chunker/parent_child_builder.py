"""
父子块构建器 — 将语义分块组织为两级层次结构

参考来源：
  - RAG项目实战 ParentDocumentRetriever（parent_splitter + child_splitter 两级设计）
  - LangChain ParentDocumentRetriever（子块搜向量库 → 返回父块完整上下文）
  - mildoc Milvus 父子双集合 Schema（parent_id 关联字段）

两阶段流程：
  1. 语义分块（semantic_splitter）→ 得到语义完整的段落块（~512 字符）
  2. 父子块构建（本模块）→ 将语义块合并为父块（~2048）+ 细分为子块（~512）

检索时的回溯逻辑（由 hybrid_searcher 实现）：
  子块做向量匹配 → child.parent_id → 取父块完整内容 → 喂给 LLM
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.chunker.semantic_splitter import SemanticChunk
from rag.config import Config


@dataclass
class ParentChunk:
    """父块：保留完整上下文的大块，直接喂给 LLM

    Attributes:
        parent_id: 业务 ID，格式 "{doc_name}_parent_{i}"
        content: 父块完整文本（目标 ~2048 字符）
        child_ids: 关联的子块 ID 列表
    """
    parent_id: str
    content: str
    child_ids: List[str] = field(default_factory=list)


@dataclass
class ChildChunk:
    """子块：用于向量精准匹配的小块

    Attributes:
        child_id: 业务 ID，格式 "{parent_id}_child_{j}"
        parent_id: 关联的父块 ID（回溯用）
        content: 子块文本（目标 ~512 字符）
    """
    child_id: str
    parent_id: str
    content: str


def _generate_parent_id(doc_name: str, index: int) -> str:
    """生成父块业务 ID。

    格式："{doc_name}_parent_{index}"
    示例："学习笔记.md_parent_0"

    Args:
        doc_name: 文档名（不含路径）
        index: 父块序号（从 0 开始）

    Returns:
        父块业务 ID 字符串
    """
    return f"{doc_name}_parent_{index}"


def _generate_child_id(parent_id: str, index: int) -> str:
    """生成子块业务 ID。

    格式："{parent_id}_child_{index}"
    示例："学习笔记.md_parent_0_child_2"

    Args:
        parent_id: 所属父块的 ID
        index: 子块在父块内的序号（从 0 开始）

    Returns:
        子块业务 ID 字符串
    """
    return f"{parent_id}_child_{index}"


def build_parent_child(
    chunks: List[SemanticChunk],
    doc_name: str,
    parent_size: int = 2048,
    parent_overlap: int = 128,
    child_size: int = 512,
    child_overlap: int = 64,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """从语义分块构建父子两级层次结构。

    构建逻辑：
      1. 遍历语义块，逐个拼接到当前父块缓冲区，直到超过 parent_size
      2. 父块确定后，用 child_splitter 将其切分为子块
      3. 子块通过 parent_id 关联回父块
      4. 最后一个父块（可能不足 parent_size）也要正常处理

    Args:
        chunks: 语义分块列表（来自 semantic_splitter.split_by_type()）
        doc_name: 文档名，用于生成 ID（如 "笔记.md"）
        parent_size: 父块目标大小（字符数），默认 2048
        parent_overlap: 父块之间的重叠字符数，默认 128
        child_size: 子块目标大小（字符数），默认 512
        child_overlap: 子块之间的重叠字符数，默认 64

    Returns:
        (parent_chunks 列表, child_chunks 列表)，两个列表通过 parent_id 关联

    Raises:
        ValueError: 如果 chunks 为空列表
    """
    # 1. 异常检查
    if not chunks:
        raise ValueError("chunks 不能为空列表")

    # 2. 准备工作
    parents = []
    children = []
    buffer = ""         # 当前正在攒的父块内容
    parent_index = 0    # 父块序号

    # 3. 创建子块切割器，可复用
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    # 4. 内部函数：将 buffer 固化为 1 个父块 + 其 N 个子块
    def _finalize_parent(buf: str, p_idx: int) -> Tuple[ParentChunk, List[ChildChunk]]:
        """把一段文本固化为一组父子块。返回 (父块, 子块列表)。"""
        # i. 生成父块业务 ID
        pid = _generate_parent_id(doc_name, p_idx)

        # ii. 用子块切割器把父块内容切成子文本
        c_texts = child_splitter.split_text(buf)

        # iii. 为每个子文本创建 ChildChunk
        c_ids = []
        c_chunks = []
        for j, ct in enumerate(c_texts):
            cid = _generate_child_id(pid, j)
            c_ids.append(cid)
            c_chunks.append(ChildChunk(child_id=cid, parent_id=pid, content=ct))

        # iv. 创建 ParentChunk，记录关联的子块 ID
        parent = ParentChunk(parent_id=pid, content=buf, child_ids=c_ids)
        return parent, c_chunks

    # 5. 遍历语义块，逐个拼接到父块缓冲区
    for chunk in chunks:
        # 5a. 加上当前 chunk 会超过 parent_size？→ 先固化现有 buffer
        if len(buffer) + len(chunk.content) > parent_size and buffer:
            parent, sub_chunks = _finalize_parent(buffer, parent_index)
            parents.append(parent)
            children.extend(sub_chunks)
            parent_index += 1
            # 5b. 重置 buffer：从旧父块末尾截取 overlap 字符，保证上下文连贯
            buffer = buffer[-parent_overlap:]

        # 5c. 将当前 chunk 拼进 buffer
        buffer += chunk.content

    # 6. 收尾：循环结束后 buffer 里还有残留内容，再做最后一次固化
    if buffer:
        parent, sub_chunks = _finalize_parent(buffer, parent_index)
        parents.append(parent)
        children.extend(sub_chunks)

    return parents, children
