"""
Office 文档解析器 — 基于 markitdown

markitdown 是微软开源的一站式文档转换库，一个库覆盖所有 Office 格式：
  - Word: .docx, .doc
  - Excel: .xlsx, .xls
  - PowerPoint: .pptx, .ppt

内部调用关系：markitdown → python-docx / openpyxl / python-pptx 等
统一接口，无需为每种格式引入不同库。
"""

import tempfile
import os
from pathlib import Path
from typing import Set

from rag.parser.base import DocumentParser


class DocxParser(DocumentParser):
    """Office 文档解析器。

    使用 markitdown 将 Word/Excel/PPT 转为 Markdown 文本。
    支持的文件类型：.docx, .doc, .xlsx, .xls, .pptx, .ppt
    """

    # 支持的所有 Office 文件扩展名
    SUPPORTED_TYPES: Set[str] = {".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt"}

    # TODO(human): 实现 __init__ 和以下方法

    def __init__(self) -> None:
        """初始化 Office 文档解析器。

        按需导入 markitdown，不在 __init__ 中加载。
        """
        # TODO(human): pass
        pass

    def supports(self, file_type: str) -> bool:
        """判断是否支持该文件类型。

        Args:
            file_type: 文件扩展名（小写），如 ".docx"

        Returns:
            True 表示 file_type 在 SUPPORTED_TYPES 集合中
        """
        # TODO(human): 判断 file_type 是否在 self.SUPPORTED_TYPES 中
        return file_type in self.SUPPORTED_TYPES

    def parse(self, data: bytes) -> str:
        """将 Office 文档字节转为 Markdown 文本。

        流程：
        1. 根据 file_type 推断正确的临时文件后缀（如 .docx → .docx）
        2. 创建临时文件，写入 data 字节
        3. 调用 markitdown.MarkItDown().convert(tmp_path) 转换
        4. 取 result.text_content 作为输出
        5. finally 清理临时文件

        Args:
            data: Office 文件的原始字节

        Returns:
            Markdown 格式的文本

        Raises:
            ImportError: markitdown 未安装
            ValueError: 文件内容无法解析
        """
        # TODO(human): 1. from markitdown import MarkItDown（按需导入）
        from markitdown import MarkItDown

        # 2. 创建临时文件（可以用 .tmp 后缀，markitdown 会自动检测格式）
        #    用 tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
        #    写入 data 字节
        tmp = tempfile.NamedTemporaryFile(suffix=".tmp", delete=False)
        try:
            tmp_path = Path(tmp.name)
            tmp_path.write_bytes(data)

            md = MarkItDown()
            result = md.convert(str(tmp_path)) # 传入文件路径字符串
        # 4. 取 result.text_content 作为输出
            return result.text_content
        # 5. finally 清理临时文件
        finally:
            self._cleanup_temp_file(tmp_path)


    @staticmethod
    def _cleanup_temp_file(path: Path) -> None:
        """安全删除临时文件（文件不存在时静默忽略）。

        Args:
            path: 临时文件的 Path 对象
        """
        # TODO(human): try: os.unlink(path) except FileNotFoundError: pass
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
