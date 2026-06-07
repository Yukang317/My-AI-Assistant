"""
语义分块器 — 按 Markdown 标题层级 + 段落边界做语义感知的文档切分

原始文档 (PDF/MD/DOCX)
    │
    ▼
  Parser（解析器，Step 5 已完成）→ 输出纯文本/Markdown 字符串
    │
    ▼
  semantic_splitter（本函数）→ 输出 List[SemanticChunk]   ← 你现在在这里
    │
    ▼
  parent_child_builder.build_parent_child() → 输出父块 + 子块
    │
    ▼
  embedding.py → 向量化
    │
    ▼
  vector_store.py → 写入 Milvus

双管道设计：
  - markdown 管道：MarkdownHeaderTextSplitter（标题切分）→ RecursiveCharacterTextSplitter（超长段落再切）
  - text 管道：RecursiveCharacterTextSplitter（中文友好分隔符降级）
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import Config


@dataclass
class SemanticChunk:
    """语义完整的文本块

    Attributes:
        content: 块文本内容
        metadata: 元数据（可能包含 headers, start_index, end_index 等）
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def split_markdown(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[SemanticChunk]:
    """对 Markdown 文本做语义分块。

    两步策略：
      1. MarkdownHeaderTextSplitter：按 H1-H6 标题层级切分，保留标题路径作为 metadata
      2. RecursiveCharacterTextSplitter：对上一步中超过 chunk_size 的段落再做递归切分

    Args:
        text: Markdown 格式的原始文本
        chunk_size: 目标块大小（字符数），默认 512
        chunk_overlap: 块之间的重叠字符数，默认 64

    Returns:
        SemanticChunk 列表，每个 chunk 的 metadata 中可能包含：
          - headers: 该块所属的标题路径（如 "H1标题 > H2标题"）
          - start_index: 在原文本中的起始位置
    """
    # 第一步：
    # MarkdownHeaderTextSplitter 的关键概念：
    #   - 它返回的每个 Document 有两个属性：.page_content（文本内容）和 .metadata（标题路径，如 {"H1": 
    #   "第一章", "H2": "1.1 概述"}）
    #   - strip_headers=False 的意思是正文里保留 ## 1.1 概述 这行标题文字
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "H1"),
            ("##", "H2"),
            ("###", "H3"),
            ("####", "H4"),
            ("#####", "H5"),
            ("######", "H6"),
        ],
        strip_headers=False,  # 保留标题行在 content 里，LLM 需要看到
    )
    # split_text 返回的是 List[Document]（LangChain 的 Document 对象）
    md_docs = md_splitter.split_text(text)

    # 第二步：
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""], # 分隔符优先级
    )

    result = []
    for doc in md_docs:
        if len(doc.page_content) <= chunk_size:
            # 标题切分后的内容没有超过 chunk_size,直接转成 SemanticChunk，metadata 继承标题信息
            result.append(SemanticChunk(
                content=doc.page_content,
                # doc.metadata 可能是这样的特殊类型：
                #   <class 'langchain_core...'>
                metadata=dict(doc.metadata),  # 标题路径
            ))
        else:
            # 超长，用text_splitter 再切
            sub_texts = text_splitter.split_text(doc.page_content)
            for sub in sub_texts:
                #   Semantic = 语义的，分割后得到带标签的语义片段
                result.append(SemanticChunk(
                    content=sub,
                    metadata=dict(doc.metadata),  # 继承父段的标题路径
                ))

    #   分块后的 result 长这样：
    #   [
    #       SemanticChunk(
    #           content="## 1.1 
    #   深度学习概述\n\n深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的表示。",
    #           metadata={"H1": "第一章", "H2": "1.1 深度学习概述"}
    #       ),
    #       SemanticChunk(
    #           content="## 1.2 卷积神经网络\n\nCNN 是一种专门处理网格结构数据的神经网络。",
    #           metadata={"H1": "第一章", "H2": "1.2 卷积神经网络"}
    #       ),
    #   ]
    return result
    

def split_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[SemanticChunk]:
    """对纯文本做递归分块（Markdown 不可用时的降级方案）。

    使用中文友好的分隔符优先级：
      ["\\n\\n", "\\n", "。", ".", " ", ""]
      段落 → 换行 → 中文句号 → 英文句号 → 空格 → 字符

    Args:
        text: 纯文本内容
        chunk_size: 目标块大小（字符数），默认 512
        chunk_overlap: 块之间的重叠字符数，默认 64

    Returns:
        SemanticChunk 列表（metadata 较简单，仅包含 start_index）
    """
    # TODO(human): 实现纯文本递归分块 — 用 RecursiveCharacterTextSplitter + 中文友好分隔符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""], # 分隔符优先级
    )
    splited_text = text_splitter.split_text(text)

    # split_text() 返回的是 List[str]，需要包装成 List[SemanticChunk]
    #   - 纯文本没有标题信息，metadata 可以用空 dict，或者记录一个 start_index
    result = []
    for i, chunk in enumerate(splited_text):
        result.append(SemanticChunk(
            content=chunk,
            metadata={"start_index": i * chunk_size},  # 记录起始位置
        ))
    
    return result


def split_by_type(
    text: str,
    doc_type: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[SemanticChunk]:
    """根据文档类型分派到对应的分块方法。

    分派规则：
      - md / markdown → split_markdown()（标题感知分块）
      - pdf / docx     → split_markdown()（pymupdf4llm / markitdown 输出已是 Markdown）
      - txt / 其他      → split_text()（纯文本降级分块）

    Args:
        text: 解析后的纯文本/Markdown 内容
        doc_type: 文档类型（pdf/docx/md/txt）
        chunk_size: 目标块大小，默认 512
        chunk_overlap: 重叠大小，默认 64

    Returns:
        SemanticChunk 列表
    """
    doc_type = doc_type.lower()


    if doc_type in ["md", "markdown", "pdf", "docx"]:
            return split_markdown(text, chunk_size, chunk_overlap)
    else:
        return split_text(text, chunk_size, chunk_overlap)

        # raise ValueError(f"Unsupported doc_type: {doc_type}")
