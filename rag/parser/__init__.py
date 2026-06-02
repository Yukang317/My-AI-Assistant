# 文档解析器：策略模式，支持 PDF / DOCX / Markdown / TXT
# 借鉴 MilDoc 的 parser/ 子包架构
#
# 使用方式：
#   from rag.parser.coordinator import DocumentParserCoordinator
#   coordinator = DocumentParserCoordinator()
#   text = coordinator.parse(file_bytes, file_type="pdf")
