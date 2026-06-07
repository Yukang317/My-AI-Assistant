# 文档分块：语义分块 + 父子块构建
# 借鉴 LlamaIndex MarkdownNodeParser + RAG项目实战的父子切分模式
#
# 流程：
#   1. semantic_splitter  → 按标题层级 + 段落边界做语义分块
#   2. parent_child_builder → 构建父子块（父块 2048/128，子块 512/64）
#
# 使用示例：
#   from rag.chunker.semantic_splitter import split_by_type
#   from rag.chunker.parent_child_builder import build_parent_child
#
#   chunks = split_by_type(text, doc_type="md")
#   parents, children = build_parent_child(chunks, doc_name="笔记.md")

from rag.chunker.semantic_splitter import SemanticChunk, split_by_type, split_markdown, split_text
from rag.chunker.parent_child_builder import ParentChunk, ChildChunk, build_parent_child

__all__ = [
    "SemanticChunk",
    "split_by_type",
    "split_markdown",
    "split_text",
    "ParentChunk",
    "ChildChunk",
    "build_parent_child",
]
