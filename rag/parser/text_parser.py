"""
纯文本解析器 — 兜底解析器

当没有其他解析器匹配时，用多编码 fallback 尝试解码。
与 MarkdownParser 使用相同的多编码策略，但作为最后一道防线。
"""

from typing import Set

from rag.parser.base import DocumentParser


class TextParser(DocumentParser):
    """纯文本兜底解析器。

    当文件类型不被其他解析器支持时，使用此解析器尝试解码。
    支持 .txt 以及任意未匹配的文件类型（作为最后的兜底）。
    """

    SUPPORTED_TYPES: Set[str] = {".txt", ".text", ".log", ".json", ".xml",
                                  ".csv", ".yaml", ".yml", ".py", ".js",
                                  ".html", ".css", ".sh", ".toml", ".ini", ".cfg"}

    # 编码尝试顺序：中文优先，最后 latin-1 兜底
    ENCODINGS: list[str] = ["utf-8", "gbk", "gb2312", "latin-1"]

    # TODO(human): 实现 __init__ 和以下方法

    def __init__(self) -> None:
        """初始化纯文本解析器。

        纯文本解码，无外部依赖，无需初始化。
        """
        # TODO(human): pass
        pass

    def supports(self, file_type: str) -> bool:
        """判断是否支持该文件类型。

        作为兜底解析器，除了 SUPPORTED_TYPES 中的已知文本格式外，
        Coordinator 会在没有其他解析器匹配时，直接将 TextParser 作为最终 fallback。

        Args:
            file_type: 文件扩展名（小写），如 ".txt"

        Returns:
            True 表示 file_type 在 SUPPORTED_TYPES 中
        """
        # TODO(human): 判断 file_type 是否在 self.SUPPORTED_TYPES 中
        return file_type in self.SUPPORTED_TYPES

    def parse(self, data: bytes) -> str:
        """用多编码 fallback 解码纯文本文件。

        流程与 MarkdownParser.parse() 相同：
        utf-8 → gbk → gb2312 → latin-1

        Args:
            data: 文本文件的原始字节

        Returns:
            解码后的文本

        Raises:
            无——latin-1 兜底保证永不抛异常
        """
        
        # TODO(human): 实现与 MarkdownParser.parse() 相同的多编码 fallback 逻辑
        # 1. 遍历 self.ENCODINGS
        # 2. try: return data.decode(encoding)
        # 3. except UnicodeDecodeError: continue
        # 4. 循环外 return data.decode("latin-1")
        for encoding in self.ENCODINGS:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        return data.decode("latin-1")
