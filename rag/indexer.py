"""
文档索引器 — 串联 Parser → Chunker → Embedding → VectorStore 的摄入管线

这是路线1（文档摄入）的顶层编排者，负责：
  1. 接收原始文件字节 → 解析 → 分块 → 向量化 → 写入 Milvus + MinIO
  2. MD5 去重：相同内容的文件自动跳过，内容更新则清理旧数据后重新索引
  3. 文档删除：一键清理三个存储系统（Milvus + MinIO + 未来 PG）

数据流（单向管道）：
  file_data (bytes)
    → ParserCoordinator.parse()
    → split_by_type()        [语义分块]
    → build_parent_child()   [父子块构建]
    → EmbeddingService.embed()
    → MilvusVectorStore.insert_parent_child()
    → MinioClient.upload()

设计原则：
  - 编排者不写业务逻辑：解析/分块/向量化/存储的具体实现都在各自模块里
  - 本模块只负责"调度的顺序"和"异常的传递"
  - 每个步骤失败时明确报错，方便定位是哪个环节出的问题
"""

import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from rag.config import Config
from rag.parser.coordinator import DocumentParserCoordinator
from rag.chunker.semantic_splitter import split_by_type
from rag.chunker.parent_child_builder import build_parent_child
from rag.embedding import EmbeddingService
from rag.storage.vector_store import MilvusVectorStore
from rag.storage.minio_client import MinioClient

logger = logging.getLogger(__name__)

# ── 模块级常量 ──────────────────────────────────────────────────────

# 文件扩展名 → MIME 类型映射（MinIO 上传时需要）
CONTENT_TYPE_MAP: Dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt":  "application/vnd.ms-powerpoint",
    ".md":   "text/markdown",
    ".txt":  "text/plain",
}


# ═══════════════════════════════════════════════════════════════════════
# 模块级工具函数
# ═══════════════════════════════════════════════════════════════════════

def _compute_md5(data: bytes) -> str:
    """计算字节数据的 MD5 哈希值。

    用于文件内容去重：两个文件即使文件名不同，只要 MD5 相同就认为是同一份文件。

    Args:
        data: 文件的原始字节内容

    Returns:
        32 位十六进制 MD5 字符串（小写），如 "d41d8cd98f00b204e9800998ecf8427e"
    """
    # TODO(human): 用 hashlib.md5(data).hexdigest() 计算并返回
    return hashlib.md5(data).hexdigest()


def _build_object_key(filename: str) -> str:
    """根据文件名构建 MinIO 对象路径。

    路径格式：documents/YYYY/MM/filename
    按年月分目录，方便在 MinIO 里按时间浏览。

    Args:
        filename: 原始文件名（如 "学习笔记.md"）

    Returns:
        MinIO object_key（如 "documents/2026/06/学习笔记.md"）
    """
    # TODO(human): 1. 用 datetime.now() 获取当前年月
    # 2. 拼接 f"documents/{year}/{month:02d}/{filename}"
    # 3. 返回路径字符串
    now = datetime.now()
    year = now.year
    month = now.month
    # 我们关心的是文件“什么时候索引的”，而不是“文件什么时候创建的”  
    # 0：用零填充（zero-padding）。2：总宽度为 2 位。d：十进制整数（decimal）
    index_path = f"documents/{year}/{month:02d}/{filename}"
    return index_path


# ═══════════════════════════════════════════════════════════════════════
# DocumentIndexer — 文档摄入编排者
# ═══════════════════════════════════════════════════════════════════════

class DocumentIndexer:
    """文档摄入编排者。

    持有四个子模块的引用，按固定顺序调度它们完成文档摄入管线。
    每个子模块各司其职，本类只负责"谁先谁后"和"中间数据怎么传递"。

    Attributes:
        parser: 文档解析协调器（策略模式，4 种解析器）
        embedder: 文本向量化服务（BGE local / API 双模式）
        vector_store: Milvus 向量存储（父子双集合）
        minio: MinIO 对象存储客户端
    """

    def __init__(self) -> None:
        """初始化所有子模块。

        四个子模块的初始化顺序无关紧要（它们之间没有依赖关系）。
        每个子模块的构造函数内部会处理自己的连接/配置（如 Milvus 连接、
        MinIO bucket 创建、BGE 模型加载等）。

        Raises:
            ConnectionError: Milvus 或 MinIO 不可达时由子模块抛出
        """
        # 1. 创建 ParserCoordinator 实例 → self.parser
        # 2. 创建 EmbeddingService 实例 → self.embedder
        # 3. 创建 MilvusVectorStore 实例 → self.vector_store
        # 4. 创建 MinioClient 实例 → self.minio
        self.parser = DocumentParserCoordinator()
        self.embedder = EmbeddingService()
        self.vector_store = MilvusVectorStore()
        self.minio = MinioClient()

    def index_document(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """完整文档摄入管线：解析 → 分块 → 向量化 → 双存储。

        这是 DocumentIndexer 的核心方法，也是对外的主要入口。
        调用方只需传入文件字节和文件名，其余全部自动处理。

        处理流程（8 步）：
          1. 计算 MD5 + 提取文件类型 + 构建 object_key
          2. MD5 去重检查：
             - 新文件（未索引过）→ 直接走完整管线
             - 内容相同：已索引且 MD5 相同 → 跳过，返回 status="skipped"
             - 内容更新：已索引但 MD5 不同 → 先删旧数据，再走正常流程
          3. ParserCoordinator.parse() → 纯文本
          4. split_by_type() → 语义块列表
          5. build_parent_child() → 父块列表 + 子块列表
          6. EmbeddingService.embed() 分别向量化父子块文本
          7. MinioClient.upload() → 原始文件存入 MinIO
          8. MilvusVectorStore.insert_parent_child() → 写入向量
             → flush() 刷盘

        Args:
            file_data: 文件的原始字节内容
            filename: 原始文件名（如 "学习笔记.md"）

        Returns:
            {
                "status": "new" | "skipped" | "updated",
                "object_key": "documents/2026/06/学习笔记.md",
                "file_md5": "d41d8cd...",
                "file_type": ".md",
                "parent_count": 3,
                "child_count": 12,
                "message": "索引完成：3 个父块，12 个子块",
            }

        Raises:
            ValueError: 文件类型不支持（不在 ALLOWED_EXTENSIONS 中）
            ValueError: 解析/分块后内容为空
            ConnectionError: Milvus 或 MinIO 不可达
            Exception: 嵌入模型加载失败
        """
        # TODO(human): 实现完整的 8 步摄入管线
        # 提示 1：file_type 从 filename 后缀提取，用 os.path.splitext()
        # 提示 2：content_type 从 CONTENT_TYPE_MAP 读取，取不到用 "application/octet-stream"
        # 提示 3：embed() 接受 List[str]，返回 List[List[float]]
        # 提示 4：调用 _prepare_chunks_for_milvus() 完成 dataclass → dict 转换
        # 提示 5：父块和子块的文本数量可能不同，embed 要分别调用
        # 提示 6：Config.ALLOWED_EXTENSIONS 检查文件类型是否允许上传的
        # 提示 7：检查向量化结果数量是否与文本数量一致
        # ========================= 第一阶段：准备工作 =========================

        # 开头加一个变量追踪文件状态
        is_update = False

        # 1.1 获取文件后缀。[0] 文件名  [1] 后缀（含点号）
        file_type = os.path.splitext(filename)[1].lower()
        if file_type not in Config.ALLOWED_EXTENSIONS:
            raise ValueError(f"文件类型 {file_type} 不支持")
        

        # 1.2 获取文件内容类型。未知类型有兜底
        content_type = CONTENT_TYPE_MAP.get(file_type, "application/octet-stream")

        # 1.3 计算文件 MD5 和构建 object_key（minio对象路径）
        file_md5 = _compute_md5(file_data)
        object_key = _build_object_key(filename)

        # 2. MD5 去重检查
        if self.vector_store.check_document_exists(object_key):
            # 取出已存在的 MD5 值
            existing_md5s = self.vector_store.get_indexed_doc_md5s()
            if existing_md5s.get(object_key) == file_md5:
                return {"status": "skipped", "object_key": object_key, "message": "文件已存在，跳过索引"}
            else:
                # 内容已改变，删除旧数据
                self.delete_document(object_key)        # 函数自定义
                is_update = True                        # 标记为更新
                # self.delete_document() 内部就会调 self.vector_store.delete_by_doc_path()
                # self.vector_store.delete_by_doc_path(object_key)  # 又删了一遍重复了
                # return {"status": "updated", "object_key": object_key, "message": "文件已更新，旧文件已删除"}  # ← 直接返回了，没走后续管线

        # 3. 解析文档。返回纯文本字符串
        text = self.parser.parse(file_data, file_type)
        if not text or not text.strip():
            raise ValueError("解析后内容为空")

        # ========================= 第二阶段：分块 + 向量化 =========================
        # 4. 语义分块。根据类型使用不同的分块策略（md/txt）
        chunks = split_by_type(text, file_type)

        # 5. 父子块构建。Tuple[List[ParentChunk], List[ChildChunk]]
        parents, children = build_parent_child(chunks, filename)

        # 6. 向量化（父子分别 embed）
        parent_texts = [p.content for p in parents]
        child_texts = [c.content for c in children]
        parent_vectors = self.embedder.embed(parent_texts)
        child_vectors = self.embedder.embed(child_texts)

        # 校验：向量数和文本数要一致
        # 个文本生成一个向量。embed() 的契约就是 List[str]  → List[List[float]]，一一对应
        if len(parent_vectors) != len(parent_texts):
            raise ValueError(f"父块向量化异常：{len(parent_texts)} 个文本 -> {len(parent_vectors)} 个向量")
        if len(child_vectors) != len(child_texts):
            raise ValueError(f"子块向量化异常：{len(child_texts)} 个文本 → {len(child_vectors)} 个向量")

        # ========================= 第三阶段： 双存储写入 =========================
        # 7. Minio 存原始文件（比 Milvus 先写，失败了不污染向量库）
        self.minio.upload(file_data, object_key, content_type)

        # 8. 数据适配 + 写入 Mlivus + 刷盘
        parent_dicts, child_dicts = self._prepare_chunks_for_milvus(
            parents, children, parent_vectors, child_vectors, 
            doc_name=filename, 
            doc_path_name=object_key, 
            doc_type=file_type, 
            doc_md5=file_md5
        )
        result = self.vector_store.insert_parent_child(parent_dicts, child_dicts)
        self.vector_store.flush()

        return {
            "status": "updated" if is_update else "new",
            "object_key": object_key,
            "file_md5": file_md5,
            "file_type": file_type,
            "parent_count": result["parent_count"],
            "child_count": result["child_count"],
            "message": f"索引完成：{result['parent_count']} 个父块，{result['child_count']} 个子块",
        }






    def _prepare_chunks_for_milvus(
        self,
        parents: List[Any],        # List[ParentChunk]
        children: List[Any],       # List[ChildChunk]
        parent_vectors: List[List[float]],
        child_vectors: List[List[float]],
        doc_name: str,
        doc_path_name: str,
        doc_type: str,
        doc_md5: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """将父子块 dataclass + 向量 组装成 Milvus 能接受的 dict 列表。

        MilvusVectorStore.insert_parent_child() 期望的 dict 格式：
          - doc_name, doc_path_name, doc_type, doc_md5
          - content（文本）, content_vector（向量）
          - embedding_model（来自 Config.get_embedding_model_name()）
          - parent_id（父块）/ parent_id + child_id（子块）

        Args:
            parents: build_parent_child 返回的 ParentChunk 列表
            children: build_parent_child 返回的 ChildChunk 列表
            parent_vectors: 与 parents 一一对应的向量列表
            child_vectors: 与 children 一一对应的向量列表
            doc_name: 文档名（用于 Milvus 字段）
            doc_path_name: MinIO object_key
            doc_type: 文件类型（如 ".md"）
            doc_md5: 文件 MD5

        Returns:
            (parent_dicts, child_dicts) — 可直接传给 insert_parent_child()

        Raises:
            ValueError: 向量数量与 chunk 数量不匹配
        """
        # TODO(human): 1. 检查 len(parents) == len(parent_vectors) 和 len(children) == len(child_vectors)
        # 2. embedding_model = Config.get_embedding_model_name()
        # 3. 遍历 parents + parent_vectors：组装父块 dict 列表
        # 4. 遍历 children + child_vectors：组装子块 dict 列表
        # 5. 返回 (parent_dicts, child_dicts)
        # 提示：ParentChunk 有 .parent_id, .content 属性
        #       ChildChunk 有 .child_id, .parent_id, .content 属性
        # 1. 校验
        if len(parents) != len(parent_vectors) or len(children) != len(child_vectors):
            raise ValueError("向量数量与 chunk 数量不匹配")
        
        # 2. 向量模型配置
        embedding_model = Config.get_embedding_model_name()

        # 3. 组装父块子块dict列表
        parent_dicts = []
        # zip() 返回一个 zip 迭代器对象，每次迭代会产生一个元组
        for chunk, vec in zip(parents, parent_vectors):
            parent_dicts.append({
                "doc_name": doc_name,
                "doc_path_name": doc_path_name,
                "doc_type": doc_type,
                "doc_md5": doc_md5,
                "embedding_model": embedding_model,
                "parent_id": chunk.parent_id,
                "content": chunk.content,
                "content_vector": vec,
            })
        
        child_dicts = []
        for chunk, vec in zip(children, child_vectors):
            child_dicts.append({
                "doc_name": doc_name,
                "doc_path_name": doc_path_name,
                "doc_type": doc_type,
                "doc_md5": doc_md5,
                "embedding_model": embedding_model,
                "parent_id": chunk.parent_id,
                "child_id": chunk.child_id,
                "content": chunk.content,
                "content_vector": vec,
            })

        return parent_dicts, child_dicts

    def delete_document(self, object_key: str) -> Dict[str, int]:
        """从 Milvus 和 MinIO 中删除一个文档的所有数据。

        执行顺序：先删 MinIO（原始文件），再删 Milvus（向量）。
        如果 MinIO 删除失败，Milvus 删除仍会执行（尽量清理干净）。

        Args:
            object_key: MinIO 对象路径（如 "documents/2026/06/学习笔记.md"）

        Returns:
            {
                "minio_deleted": 1,
                "parent_deleted": 3,
                "child_deleted": 12,
            }

        Raises:
            ValueError: object_key 为空时抛出
        """
        # TODO(human): 1. 检查 object_key 非空
        # 2. MinIO: self.minio.delete(object_key)  # 文件不存在时静默忽略
        # 3. Milvus: self.vector_store.delete_by_doc_path(object_key)
        # 4. 返回删除计数
        if not object_key:
            raise ValueError("object_key 为空")

        # MinIO 删除 — 文件不存在时静默忽略
        # minio_deleted_result = self.minio.delete(object_key)
        self.minio.delete(object_key)

        
        # Milvus 删除 — 尽力清理
        # 返回 {"parent_deleted": N, "child_deleted": M}
        vector_deletion_result = self.vector_store.delete_by_doc_path(object_key)
        return {
            # "minio_deleted": minio_deleted_result + "minio是静默删除应该不会返回值",
            "minio_deleted": 1,
            "parent_deleted": vector_deletion_result["parent_deleted"],
            "child_deleted": vector_deletion_result["child_deleted"],
        }
