"""
文档解析器抽象基类

所有解析器必须实现 parse() 和 supports() 两个方法。
借鉴 MilDoc 的 DocumentParser 抽象设计。
"""

from abc import ABC, abstractmethod


class DocumentParser(ABC):
    """文档解析器抽象基类。

    每个子类负责解析一种文件类型，输入文件字节，输出纯文本。
    策略模式：Coordinator 根据 file_type 自动选择匹配的解析器。
    """

    @abstractmethod
    def parse(self, data: bytes) -> str:
        """将文件字节解析为纯文本。

        Args:
            data: 文件的原始字节内容

        Returns:
            解析后的纯文本字符串

        Raises:
            ValueError: 文件内容无法解析时抛出
        """
        ...

    @abstractmethod
    def supports(self, file_type: str) -> bool:
        """判断该解析器是否支持给定的文件类型。

        Args:
            file_type: 文件扩展名（小写，含点号），如 ".pdf"、".docx"

        Returns:
            True 表示该解析器能处理此文件类型
        """
        ...


# ============================================================================
# 设计说明
# ============================================================================
#
# 1. ABC (Abstract Base Class)
#    - 定义接口规范：强制子类必须实现某些方法
#    - 防止直接实例化：不能直接创建 DocumentParser() 对象，只能创建其子类
#    - 提供类型检查：确保所有解析器都遵循相同的接口
#
# 2. @abstractmethod 装饰器
#    - 强制实现：任何继承 DocumentParser 的子类都必须实现这两个方法
#    - 运行时检查：如果子类没有实现这些方法，Python 会在实例化时报错
#    - 保证一致性：确保所有解析器都有统一的接口
#
# 3. 策略模式实现
#    - 不同的文件类型（PDF、Word、TXT等）需要不同的解析逻辑
#    - 但对外提供统一的接口（parse 和 supports）
#    - Coordinator 可以根据文件类型自动选择合适的解析器
#
