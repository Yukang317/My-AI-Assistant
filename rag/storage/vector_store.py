"""
Milvus 向量存储封装 — 父子双 Collection

核心功能：
  1. 文档摄入：将父子分块向量写入 Milvus（parent_chunks + child_chunks）
  2. 知识检索：子集合语义搜索 → 通过 parent_id 回溯父块获取完整上下文
  3. 文档管理：支持按路径删除、存在 去重
  4. 生命周期：flush 刷盘、load/release 内存管理

技术架构：
  - 双 Collection 设计：child_chunks(512 tokens) 用于精准检索，parent_chunks(2048 tokens) 提供完整上下文
  - 索引策略：IVF_FLAT + IP（内积），适配 BGE 归一化向量
  - 检索链路：query_vector → child_collection.search() → parent_id 去重 → parent_collection.query()
  - 字段设计：doc_path_name 作为唯一标识，支持增量更新和删除

使用场景：
  - RAG 系统文档索引：parse → chunk → embed → insert_parent_child()
  - 智能问答检索：embed(query) → search() → 返回父子块组合结果
  - 文档更新：check_document_exists() → delete_by_doc_path() → 重新索引

依赖服务：
  - Milvus 向量数据库（HTTP 连接）
  - BGE embedding 模型（1024 维向量）
"""

import logging
from typing import Any, Dict, List, Optional
import json

from pymilvus import DataType, MilvusClient

from rag.config import Config

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus 向量存储 — 管理 parent_chunks 和 child_chunks 两个 Collection。

    每个 Collection 包含 Schema 字段（id, doc_name, doc_path_name, doc_type,
    doc_md5, parent_id/child_id, content, content_vector, embedding_model）
    以及 IVF_FLAT + IP 索引。

    Attributes:
        client: MilvusClient 实例（HTTP 连接，非 gRPC）
        parent_collection: 父集合名称（来自 Config）
        child_collection: 子集合名称（来自 Config）
        parent_schema: 父集合的 CollectionSchema（_init_collections 后可用）
        child_schema: 子集合的 CollectionSchema（_init_collections 后可用）
    """

    # ── 字段长度常量 ──────────────────────────────────────────────
    MAX_DOC_NAME_LEN = 500       # 文档名最大长度
    MAX_DOC_PATH_LEN = 1000      # MinIO object_key 最大长度
    MAX_DOC_TYPE_LEN = 50        # 文件类型（pdf/docx/md/txt）
    MAX_BUSINESS_ID_LEN = 200    # parent_id / child_id 最大长度
    MAX_CONTENT_LEN = 65535      # VARCHAR 上限（Milvus 限制）
    MAX_EMBEDDING_MODEL_LEN = 100  # 模型名最大长度

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """初始化 Milvus 客户端并创建/加载 Collection。

        连接 Milvus 后调用 _init_collections() 确保两个 Collection
        和索引存在（幂等操作）。同时将 Schema 存入实例变量供后续复用。

        Args:
            host: Milvus 服务地址，默认从 Config.MILVUS_HOST 读取
            port: Milvus 服务端口，默认从 Config.MILVUS_PORT 读取
            db_name: 数据库名，默认从 Config 推断（当前未配置则用 "default"）

        Raises:
            ConnectionError: Milvus 服务不可达时抛出
        """
        # TODO(human): 用 MilvusClient 连接 Milvus → 存 self.client → 调 _init_collections()
        # 提示：host/port 用传入参数或 Config 默认值，db_name 用传入或默认 "default"
        # _init_collections() 会设置 self.parent_schema 和 self.child_schema
        host = host or Config.MILVUS_HOST
        port = port or Config.MILVUS_PORT
        db_name = db_name or Config.MILVUS_DB_NAME

        try:
            # pymilvus 用 uri 参数，不是分开的 host/port
            self.client = MilvusClient(uri=f"http://{host}:{port}", db_name=db_name)
            self._init_collections()
        except Exception as e:
            raise ConnectionError(f"无法连接到 Milvus: {e}")



    # ═══════════════════════════════════════════════════════════════
    # Schema 构建
    # ═══════════════════════════════════════════════════════════════

    def _build_parent_schema(self) -> Any:
        """构建 parent_chunks Collection 的 Schema。

        parent_chunks 字段（9 个）：
          - id: INT64 主键，自增（auto_id=True）
          - doc_name: VARCHAR(500)
          - doc_path_name: VARCHAR(1000)，MinIO object_key，用于删除/更新的唯一标识
          - doc_type: VARCHAR(50)，文件类型
          - doc_md5: VARCHAR(32)，文件 MD5 去重
          - parent_id: VARCHAR(200)，业务 ID: "{doc_name}_parent_{i}"
          - content: VARCHAR(65535)，父块完整文本
          - content_vector: FLOAT_VECTOR(1024)，BGE 向量
          - embedding_model: VARCHAR(100)，记录向量生成模型

        Returns:
            pymilvus.CollectionSchema 对象（尚未创建，仅定义结构）

        Raises:
            ValueError: EMBEDDING_DIM 不是正数时抛出
        """
        # TODO(human): 1. 用 MilvusClient.create_schema(auto_id=True) 创建 schema
        # 2. schema.add_field("id", DataType.INT64, is_primary=True)
        # 3. 依次添加 8 个字段（上面列出的，跳过 id）
        # 4. 返回 schema
        # 提示：content_vector 的 dim 从 Config.EMBEDDING_DIM 读取
        parent_schema = MilvusClient.create_schema(auto_id = True)
        parent_schema.add_field("id", DataType.INT64, is_primary=True)
        parent_schema.add_field("doc_name", DataType.VARCHAR, max_length=self.MAX_DOC_NAME_LEN)
        parent_schema.add_field("doc_path_name", DataType.VARCHAR, max_length=self.MAX_DOC_PATH_LEN)
        parent_schema.add_field("doc_type", DataType.VARCHAR, max_length=self.MAX_DOC_TYPE_LEN)
        parent_schema.add_field("doc_md5", DataType.VARCHAR, max_length=32)
        parent_schema.add_field("parent_id", DataType.VARCHAR, max_length=self.MAX_BUSINESS_ID_LEN)
        parent_schema.add_field("content", DataType.VARCHAR, max_length=self.MAX_CONTENT_LEN)
        parent_schema.add_field("content_vector", DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        parent_schema.add_field("embedding_model", DataType.VARCHAR, max_length=self.MAX_EMBEDDING_MODEL_LEN)
        return parent_schema


    def _build_child_schema(self) -> Any:
        """构建 child_chunks Collection 的 Schema。

        child_chunks 字段（9 个）：
          - id: INT64 主键，自增
          - doc_name: VARCHAR(500)
          - doc_path_name: VARCHAR(1000)，MinIO object_key
          - parent_id: VARCHAR(200)，关联父块的业务 ID（检索后回溯用）
          - child_id: VARCHAR(200)，业务 ID: "{parent_id}_child_{j}"
          - doc_type: VARCHAR(50)，文件类型，支持搜索时按类型过滤
          - content: VARCHAR(65535)，子块文本（512 tokens）
          - content_vector: FLOAT_VECTOR(1024)
          - embedding_model: VARCHAR(100)

        Returns:
            pymilvus.CollectionSchema 对象

        Raises:
            ValueError: EMBEDDING_DIM 不是正数时抛出
        """
        # TODO(human): 与 _build_parent_schema 结构类似，但字段略有不同
        # 注意：用 child_id 替代 parent_id（child_id 才是 child 的业务主键）
        # parent_id 字段依然保留，用于回溯父块
        child_schema = MilvusClient.create_schema(auto_id = True)
        child_schema.add_field("id", DataType.INT64, is_primary=True)
        child_schema.add_field("doc_name", DataType.VARCHAR, max_length=self.MAX_DOC_NAME_LEN)
        child_schema.add_field("doc_path_name", DataType.VARCHAR, max_length=self.MAX_DOC_PATH_LEN)
        child_schema.add_field("parent_id", DataType.VARCHAR, max_length=self.MAX_BUSINESS_ID_LEN)
        child_schema.add_field("child_id", DataType.VARCHAR, max_length=self.MAX_BUSINESS_ID_LEN)
        child_schema.add_field("doc_type", DataType.VARCHAR, max_length=self.MAX_DOC_TYPE_LEN)
        child_schema.add_field("content", DataType.VARCHAR, max_length=self.MAX_CONTENT_LEN)
        child_schema.add_field("content_vector", DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        child_schema.add_field("embedding_model", DataType.VARCHAR, max_length=self.MAX_EMBEDDING_MODEL_LEN)
        return child_schema


    # ═══════════════════════════════════════════════════════════════
    # Collection 初始化
    # ═══════════════════════════════════════════════════════════════

    def _init_collections(self) -> None:
        """初始化父子两个 Collection（幂等操作）。

        流程：
        1. 构建两个 Schema → 存入 self.parent_schema / self.child_schema
        2. 对每个 Collection：不存在则创建 → 创建索引 → 加载到内存
        3. 已存在的 Collection：只做 load（不重复创建）

        设计要点：
        - 幂等：多次调用不会报错或重复创建
        - 自动 load：初始化后即可搜索（不需要手动调 load_collection）
        """
        self.parent_schema = self._build_parent_schema()
        self.child_schema = self._build_child_schema()
        self._create_collection_if_not_exists(Config.MILVUS_PARENT_COLLECTION, self.parent_schema)
        self._create_collection_if_not_exists(Config.MILVUS_CHILD_COLLECTION, self.child_schema)


    def _create_collection_if_not_exists(
        self, collection_name: str, schema: Any
    ) -> None:
        """检查 Collection 是否存在，不存在则创建 + 建索引 + 加载。

        完整流程（参考 mildoc 生产实践）：
          has_collection() → 不存在 → create_collection() → create_index() → load_collection()
          has_collection() → 存在   → load_collection()（若未加载）

        Args:
            collection_name: Collection 名称（如 "parent_chunks"）
            schema: pymilvus CollectionSchema 对象

        Raises:
            ConnectionError: Milvus 不可达
        """
        # TODO(human): 1. 用 self.client.has_collection(collection_name) 检查
        # 2. 不存在 → self.client.create_collection(collection_name, schema=schema)
        #    → self._create_index_if_not_exists(collection_name)
        # 3. self.client.load_collection(collection_name)  # 加载到内存
        # 提示：如果已存在，直接 load 即可（可能已经 load 了，再 load 一次无害）
        if not self.client.has_collection(collection_name):
            self.client.create_collection(collection_name, schema=schema)
            self._create_index_if_not_exists(collection_name)
        self.client.load_collection(collection_name)





    def _create_index_if_not_exists(self, collection_name: str) -> None:
        """为 Collection 的 content_vector 字段创建 IVF_FLAT 索引。

        索引参数（从 Config 读取）：
          - index_type: IVF_FLAT
          - metric_type: IP（内积，因为 BGE 已做 L2 归一化）
          - params: {"nlist": 1024}

        使用 self.client.list_indexes() 检查是否已有索引，避免重复创建。

        Args:
            collection_name: 要建索引的 Collection 名称

        Raises:
            ConnectionError: Milvus 不可达
        """

        # self.client.list_indexes(collection_name)
        if not self.client.has_index(collection_name, "content_vector"):
            self.client.create_index(
                collection_name, field_name="content_vector",
                index_type=Config.MILVUS_INDEX_TYPE,
                metric_type=Config.MILVUS_METRIC_TYPE,
                params={"nlist": Config.MILVUS_INDEX_NLIST}
            )




    # ═══════════════════════════════════════════════════════════════
    # 数据写入
    # ═══════════════════════════════════════════════════════════════

    def insert_parent_child(self, parent_chunks: List[Dict[str, Any]], child_chunks: List[Dict[str, Any]],) -> Dict[str, int]:
        """分别向父子 Collection 批量写入数据。

        每个 chunk 字典必须包含：
          - doc_name: 来源文档名
          - doc_path_name: MinIO object_key
          - doc_type: 文件类型
          - doc_md5: 文件 MD5
          - content: 文本内容
          - content_vector: BGE 向量（List[float]）
          - embedding_model: 模型名
          - parent_id: 业务 ID（parent 必需；child 额外需要 child_id）

        Args:
            parent_chunks: 父块列表，每个元素含 content_vector 等字段
            child_chunks: 子块列表，每个元素额外含 parent_id + child_id

        Returns:
            {"parent_count": N, "child_count": M}，分别记录插入条数

        Raises:
            ValueError: 列表为空或缺少必填字段
            ConnectionError: Milvus 不可达
        """
        # 1. 检查非空
        if not parent_chunks:
            raise ValueError("parent_chunks 列表为空，无法插入")
        if not child_chunks:
            raise ValueError("child_chunks 列表为空，无法插入")
        
        # 2. 数据插入到集合
        parent_result = self.client.insert(
            collection_name=Config.MILVUS_PARENT_COLLECTION,
            data=parent_chunks
        )
        child_result = self.client.insert(
            collection_name=Config.MILVUS_CHILD_COLLECTION,
            data=child_chunks
        )
        # xxxx_result的内容，pymvius会返回这个格式的字典。
        # {
        #     "insert_count": 3,        # ← 实际插入了几条
        #     "ids": [1, 2, 3],         # ← Milvus 自动分配的主键 ID
        # }


        # 3. 返回条数
        return {
            "parent_count": parent_result["insert_count"],
            "child_count": child_result["insert_count"]
        }



    # ═══════════════════════════════════════════════════════════════
    # 向量检索
    # ═══════════════════════════════════════════════════════════════

    def search(self, query_vector: List[float], top_k: int = 10, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """在 child_collection 中检索 → 回溯 parent_collection 取完整父块。

        这是整个检索链路的入口：子块做语义匹配（精度高）→ 通过 parent_id
        回父集合取完整上下文。

        Args:
            query_vector: 查询向量（已做 L2 归一化，维度与 向量数据库中 Collection 一致）
            top_k: 返回的最相似子块数量，默认 10
            filter_expr: Milvus 过滤表达式，如：
                'doc_type == "md"' 或 'embedding_model == "bge-large-zh-v1.5"'
                传 None 则不添加额外过滤条件

        Returns:
            [
                {
                    "child_id": "...",
                    "parent_id": "...",
                    "child_content": "子块文本（512 tokens）",
                    "parent_content": "父块完整文本（2048 tokens）",
                    "doc_name": "来源文档名",
                    "doc_path_name": "MinIO object_key",
                    "doc_type": "md",
                    "score": 0.92,  # IP 距离（越高越相似）
                },
                ...
            ]
            按 score 降序排列

        Raises:
            ValueError: query_vector 维度不匹配
            ConnectionError: Milvus 不可达
        """
        search_params = {"nprobe": Config.MILVUS_SEARCH_NPROBE}
        # 默认过滤条件：只搜索当前 embedding 模型的向量
        # 如果调用方提供了额外过滤条件，则拼接上去
        expr = f'embedding_model == "{Config.get_embedding_model_name()}"'
        if filter_expr:
            expr += f" and {filter_expr}"
        
        results = self.client.search(
            collection_name=Config.MILVUS_CHILD_COLLECTION,
            data=[query_vector],
            anns_field="content_vector",    # ANN要搜索的字段
            limit=top_k,
            filter=expr,        # 搜索前过滤
            search_params=search_params, # IVF_FLAT索引搜索算法的参数配置
            output_fields=["child_id", "parent_id", "content", "doc_name",
                           "doc_path_name", "doc_type"],
        )
        # 每个命中（hit）有三个关键字段：（results内容）
        #   - "id"：Milvus 自动分配的主键
        #   - "distance"：相似度分数（IP 内积值，越大越相似）
        #   - "entity"：我们在 output_fields 中指定的字段

        # results[0] 可能是空的——一个命中都没有。这时候直接返回空列表 []，不用继续查父块
        if not results[0]:
            return []

        parent_ids = []
        # 因为只有一个查询向量，所以把所有命中的 parent_id 收集起来
        for hit in results[0]:
            parent_ids.append(hit["entity"]["parent_id"])
        
        # 去重
        parent_ids = list(set(parent_ids))
        parent_ids_str = json.dumps(parent_ids) # 自动处理引号

        # 返回一个列表
        parents = self.client.query(
            collection_name=Config.MILVUS_PARENT_COLLECTION,
            # filter 里用 in 的好处：一次查询就把所有父块都取回来了，不用循环发 N 次请求。
            filter=f'parent_id in {parent_ids_str}',
            output_fields=["parent_id", "content"],
        )

        # 先把父块查出来的结果转成{parent_id: content} 字典，方便查找
        parent_content_map = {}
        for p in parents:
            parent_content_map[p["parent_id"]] = p["content"]

        # 组装返回，同时对同一 parent_id 去重（只保留 score 最高那条）
        dedup = {}
        for hit in results[0]:
            pid = hit["entity"]["parent_id"]
            score = hit["distance"]

            # 同一个 parent_id 只保留分数最高的
            if pid not in dedup or score > dedup[pid]["score"]:
                dedup[pid] = {
                    "parent_id": pid,                                   # 
                    "child_id": hit["entity"]["child_id"], 
                    "child_content": hit["entity"]["content"],
                    "parent_content": parent_content_map.get(pid, ""),  # 从 map 里取父块的内容
                    "doc_name": hit["entity"]["doc_name"],
                    "doc_path_name": hit["entity"]["doc_path_name"],
                    "doc_type": hit["entity"]["doc_type"],
                    "score": score,                           # 相似度分数（IP 内积值，越大越相似）
                }
        # dedup的内容
        #   {
        #       "报告_A_p0": {
        #           "child_id": "报告_A_p0_c3",
        #           "parent_id": "报告_A_p0",
        #           "child_content": "子块 c3 的文本...",
        #           "parent_content": "父块 p0 的完整文本（2048 tokens）...",
        #           "doc_name": "Q4报告.md",
        #           "doc_path_name": "documents/2024/Q4报告.md",
        #           "doc_type": "md",
        #           "score": 0.92,
        #       },
        #       "报告_B_p0": {
        #           "child_id": "报告_B_p0_c5",
        #           "parent_id": "报告_B_p0",
        #           "child_content": "子块 c5 的文本...",
        #           "parent_content": "父块 p0 的完整文本...",
        #           "doc_name": "周报.md",
        #           "doc_path_name": "documents/2024/周报.md",
        #           "doc_type": "md",
        #           "score": 0.87,
        #       },
        #   }

        output = list(dedup.values())
        # 按 score 降序排列（search 默认降序，这里保险一下）
        output.sort(key=lambda x: x["score"], reverse=True)
        return output
        



        





    # ═══════════════════════════════════════════════════════════════
    # 文档删除与检查
    # ═══════════════════════════════════════════════════════════════

    def delete_by_doc_path(self, doc_path_name: str) -> Dict[str, int]:
        """删除指定文档在父子 Collection 中的所有向量。

        用于：文档内容更新时清理旧向量；文档删除时清理 Milvus 数据。

        安全检查（来自 mildoc 的 _validate_path）：
          拒绝空字符串或空白字符串，防止误删全库。

        Args:
            doc_path_name: MinIO object_key（如 "documents/2024/notes.md"）

        Returns:
            {"parent_deleted": N, "child_deleted": M}

        Raises:
            ValueError: doc_path_name 为空或空白
            ConnectionError: Milvus 不可达
        """
        # TODO(human): 1. 安全检查：if not doc_path_name or not doc_path_name.strip() → raise ValueError
        # 2. 构建 filter_expr: f'doc_path_name == "{doc_path_name}"'
        # 3. 从 parent_collection 和 child_collection 分别删除
        #    self.client.delete(collection_name, filter=expr)
        # 4. 返回删除数量
        if not doc_path_name or not doc_path_name.strip():
            raise ValueError("doc_path_name 为空或空白")
        
        filter_expr = f'doc_path_name == "{doc_path_name}"'
        parent_result = self.client.delete(
            collection_name=Config.MILVUS_PARENT_COLLECTION, filter=filter_expr
        )
        child_result = self.client.delete(
            collection_name=Config.MILVUS_CHILD_COLLECTION, filter=filter_expr
        )
        # delete() 返回 {"delete_count": N}，和 insert() 返回格式一致
        return {
            "parent_deleted": parent_result["delete_count"],
            "child_deleted": child_result["delete_count"],
        }



    def check_document_exists(self, doc_path_name: str) -> bool:
        """检查文档是否已在 Milvus 中索引（用于去重判断）。

        Args:
            doc_path_name: MinIO object_key

        Returns:
            True 表示至少有一条向量记录属于该文档

        Raises:
            ValueError: doc_path_name 为空
        """
        # TODO(human): 1. 在 child_collection 中 query: filter=f'doc_path_name == "{doc_path_name}"'
        # 2. limit=1, output_fields=["id"]（只要 id，不传输大字段）
        # 3. 返回 len(results) > 0
        if not doc_path_name:
            raise ValueError("doc_path_name 为空")
        
        filter_expr = f'doc_path_name == "{doc_path_name}"'
        results = self.client.query(
            collection_name=Config.MILVUS_CHILD_COLLECTION,
            filter=filter_expr,
            output_fields=["id"],
            limit=1,
        )
        return len(results) > 0

    def get_indexed_doc_md5s(self) -> Dict[str, str]:
        """获取所有已索引文档的 MD5 映射。

        Returns:
            {doc_path_name: doc_md5} — 用于增量索引时对比 MD5 是否变化

        Raises:
            ConnectionError: Milvus 不可达
        """
        # TODO(human): 1. 从 parent_collection query 所有记录：
        #    output_fields=["doc_path_name", "doc_md5"]
        #    注意 limit 可能受限，数据量大时需要分页或增大 limit
        # 2. 组装成 {row["doc_path_name"]: row["doc_md5"]} 字典
        # 3. 返回字典
        results = self.client.query(
            collection_name=Config.MILVUS_PARENT_COLLECTION,
            output_fields=["doc_path_name", "doc_md5"],
            limit=1000,  # 假设一次查询 1000 条足够
        )
        return {row["doc_path_name"]: row["doc_md5"] for row in results}

    # ═══════════════════════════════════════════════════════════════
    # 生命周期管理（参考 mildoc 的 flush/load/release 模式）
    # ═══════════════════════════════════════════════════════════════

    def flush(self) -> None:
        """强制刷盘 — 将内存中的未持久化数据写入磁盘。

        调用时机：索引完成后调用，确保 Milvus 数据不会因意外重启丢失。
        对两个 Collection 分别执行 flush。

        Raises:
            ConnectionError: Milvus 不可达
        """
        # TODO(human): self.client.flush([Config.MILVUS_PARENT_COLLECTION, Config.MILVUS_CHILD_COLLECTION])
        self.client.flush([Config.MILVUS_PARENT_COLLECTION, Config.MILVUS_CHILD_COLLECTION])


        
    def load_collection(self) -> None:
        """将两个 Collection 加载到内存（搜索前必须加载）。

        调用时机：启动服务时、release 后需要搜索前。
        已加载时重复调用无害。

        Raises:
            ConnectionError: Milvus 不可达
        """
        # TODO(human): 对两个 Collection 分别 load_collection

        self.client.load_collection(Config.MILVUS_PARENT_COLLECTION)
        self.client.load_collection(Config.MILVUS_CHILD_COLLECTION)

    def release_collection(self) -> None:
        """释放 Collection 内存（省 ~200MB）。

        调用时机：索引完成后（日常聊天不需要 Milvus 驻留内存）。
        ECS 只有 3.5GB 内存，BGE 模型 + Milvus 同时驻留容易 OOM。

        Raises:
            ConnectionError: Milvus 不可达
        """
        # TODO(human): 对两个 Collection 分别 release_collection
        self.client.release_collection(Config.MILVUS_PARENT_COLLECTION)
        self.client.release_collection(Config.MILVUS_CHILD_COLLECTION)
