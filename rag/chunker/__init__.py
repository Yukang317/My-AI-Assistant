# 文档分块：语义分块 + 父子块构建
# 借鉴 LlamaIndex MarkdownNodeParser + RAG项目实战的父子切分模式
#
# 流程：
#   1. semantic_splitter  → 按标题层级 + 段落边界做语义分块
#   2. parent_child_builder → 构建父子块（父块 2048/128，子块 512/64）
