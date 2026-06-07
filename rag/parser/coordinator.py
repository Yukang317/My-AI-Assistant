"""
文档解析协调器 — 策略模式入口

遍历注册的解析器列表，找到第一个 supports() 返回 True 的解析器，
调用其 parse() 方法。如果没有匹配的解析器，使用 TextParser 兜底。

借鉴 MilDoc 的 SimpleObjectParser 架构，但简化了嵌套策略模式。
平面注册 + 优先级排序，意图识别是后续 Agent 的事。
"""

from typing import List

from rag.parser.base import DocumentParser
from rag.parser.pdf_parser import PDFParser
from rag.parser.docx_parser import DocxParser
from rag.parser.markdown_parser import MarkdownParser
from rag.parser.text_parser import TextParser


class DocumentParserCoordinator:
    """文档解析协调器。

    职责：
    1. 注册所有解析器实例（按优先级排列）
    2. 根据 file_type 找到匹配的解析器
    3. 调用解析器的 parse() 方法

    不负责：
    - MinIO 文件读写（调用方负责提供 bytes）
    - 分块/向量化（chunker/vector_store 负责）
    - 意图识别（后续 Agent 负责）

    Attributes:
        parsers: 按优先级排列的解析器列表，兜底解析器放最后
    """

    # TODO(human): 实现 __init__、_get_parser、parse、supported_types 四个方法

    def __init__(self) -> None:
        """初始化协调器，注册所有解析器。

        解析器按优先级排列：
        1. PDFParser      — 处理 .pdf
        2. DocxParser     — 处理 .docx/.doc/.xlsx/.xls/.pptx/.ppt
        3. MarkdownParser — 处理 .md/.markdown
        4. TextParser     — 处理 .txt 等纯文本 + 兜底

        解析器实例按需创建（不在 __init__ 中加载任何重型库）。
        """
        # TODO(human): 按优先级创建解析器列表，TextParser 放最后做兜底
        self.parsers = [
            PDFParser(),
            DocxParser(),
            MarkdownParser(),
            TextParser(),
        ]

    def _get_parser(self, file_type: str) -> DocumentParser:
        """根据文件类型找到匹配的解析器。

        遍历 self.parsers，返回第一个 supports(file_type) 返回 True 的解析器。
        因为 TextParser 放在列表最后且 supports() 很宽泛，
        所以任何文件类型至少会被 TextParser 兜底。

        Args:
            file_type: 文件扩展名（小写，含点号），如 ".pdf"

        Returns:
            匹配的 DocumentParser 实例（永不返回 None，TextParser 兜底）
        """
        # TODO(human): 1. 遍历 self.parsers
        # 2. if parser.supports(file_type): return parser
        # 3. 循环外 return self.parsers[-1]  # TextParser 兜底
        for parser in self.parsers:
            if parser.supports(file_type):
                return parser

        # TextParser 兜底
        return self.parsers[-1]

    def parse(self, data: bytes, file_type: str) -> str:
        """解析文件内容。

        这是对外的唯一入口，调用方只需提供文件字节和扩展名。

        Args:
            data: 文件的原始字节内容
            file_type: 文件扩展名（小写），如 ".pdf"、".docx"

        Returns:
            解析后的纯文本字符串

        Raises:
            ValueError: 解析失败时抛出
            FileNotFoundError: 临时文件写入失败时抛出（极少见）
        """
        # TODO(human): 1. 调 _get_parser(file_type) 找解析器
        # 2. return parser.parse(data)
        parser = self._get_parser(file_type)
        return parser.parse(data)

    def supported_types(self) -> List[str]:
        """返回所有解析器支持的文件类型汇总。

        Returns:
            所有支持的文件扩展名列表，如 [".pdf", ".docx", ".doc", ...]
        """
        # TODO(human): 1. 遍历 self.parsers
        # 2. 收集每个 parser 的 SUPPORTED_TYPES（或 supports() 方法）
        #    提示：PDFParser 可能没有 SUPPORTED_TYPES 属性，用 parser.supports() 来判断的话需要知道所有可能类型
        #    推荐做法：让每个解析器都有 SUPPORTED_TYPES 类属性，然后汇总
        # 3. 返回去重排序后的列表
        types = []
        for parser in self.parsers:
            if hasattr(parser, "SUPPORTED_TYPES"):
                types.extend(parser.SUPPORTED_TYPES)

        return sorted(set(types))
