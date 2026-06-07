"""
PDF 文档解析器 — 基于 pymupdf4llm

pymupdf4llm 将 PDF 直接转换为 Markdown 格式，保留：
  - 标题层级（H1-H6）
  - 表格结构
  - 图片引用（提取到独立目录）
  - 公式（LaTeX 格式）

比 PyPDF2 强 10 倍，是 LlamaIndex RAG 项目的首选 PDF 方案。
"""

import tempfile
import os
from pathlib import Path
from typing import Set

from rag.parser.base import DocumentParser


class PDFParser(DocumentParser):
    """PDF 文件解析器。

    使用 pymupdf4llm 将 PDF 转为 Markdown 文本。
    支持的文件类型：.pdf
    """
    SUPPORTED_TYPES = {".pdf"}

    # TODO(human): 实现 __init__ 和以下方法

    def __init__(self) -> None:
        """初始化 PDF 解析器。

        注意：pymupdf4llm 依赖 PyMuPDF，首次导入可能较慢（~1秒）。
        这里不做任何初始化，按需导入在 parse() 中完成。
        """
        # TODO(human): pass（暂时不需要初始化任何东西）
        pass

    def supports(self, file_type: str) -> bool:
        """判断是否支持该文件类型。

        Args:
            file_type: 文件扩展名（小写），如 ".pdf"

        Returns:
            True 表示 file_type == ".pdf"
        """
        # TODO(human): 判断 file_type 是否为 ".pdf"
        return file_type == ".pdf"

    def parse(self, data: bytes) -> str:
        """将 PDF 字节转为 Markdown 文本。

        流程：
        1. 创建临时文件（.pdf 后缀），写入 data 字节
        2. 调用 pymupdf4llm.to_markdown(tmp_path) 转换
        3. 删除临时文件（finally 保证清理）
        4. 返回 Markdown 文本

        Args:
            data: PDF 文件的原始字节

        Returns:
            Markdown 格式的文本（含标题/表格/图片引用）

        Raises:
            ImportError: pymupdf4llm 未安装
            ValueError: PDF 内容无法解析
        """
        # TODO(human): 1. import pymupdf4llm（按需导入，仅在 parse 时加载）
        # 2. 用 tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) 创建临时文件
        # 3. tmp_path.write_bytes(data)  # Path 对象写字节
        # 4. 调 pymupdf4llm.to_markdown(doc=tmp_path) 转换
        # 5. finally 块中 os.unlink(tmp_path) 清理临时文件
        # 6. 返回 md_text
        import pymupdf4llm
        # 文件关闭时自动删除，后缀.pdf。
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            # NamedTemporaryFile 返回的对象有.name 属性（文件路径字符串），用 Path(tmp.name) 包装后就能用 Path 的所有方法
            tmp_path = Path(tmp.name)
            # 内部会 open("wb") → write → close，一行搞定
            tmp_path.write_bytes(data)

            return pymupdf4llm.to_markdown(doc=tmp_path)
        finally:
            self._cleanup_temp_file(tmp_path)

        

    # 不需要访问 self 的任何属性，只是一个工具函数
    @staticmethod
    def _cleanup_temp_file(path: Path) -> None:
        """安全删除临时文件（文件不存在时静默忽略）。

        Args:
            path: 临时文件的 Path 对象
        """
        # TODO(human): try: os.unlink(path) except FileNotFoundError: pass
        # os.unlink() 删除不存在的文件会抛 FileNotFoundError。在finally 块中，如果 pymupdf4llm.to_markdown()
        #    内部已经清理了临时文件（某些版本会这样做），我们再次 unlink 就会出错
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
