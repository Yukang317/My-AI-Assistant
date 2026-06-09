from typing import List
from rag.config import Config


class EmbeddingService:
    """
    文本向量化服务
    - local 模式：加载 BGE 模型到内存（SentenceTransformer）
    - api 模式：调用远程 embedding API
    """
    def __init__(self, mode: str = Config.EMBEDDING_MODE):
        self.mode = mode
        self.model = None # local 模式的模型实例
        self.client = None # api 模式的 OpenAI client

        if mode == "local":
            self._init_local_model()  # 私有方法
        elif mode == "api":
            self._init_api_client()
        else:
            raise ValueError(f"不支持的 embedding 模式：{mode}")

    # 本地模式
    def _init_local_model(self):
        """加载 BGE 模型到内存"""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name_or_path = Config.EMBEDDING_LOCAL_MODEL,
            device = "cpu"
        )

    # API 模式
    def _init_api_client(self):
        """初始化 OpenAI 兼容的embedding 客户端"""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=Config.EMBEDDING_API_KEY,
            base_url=Config.EMBEDDING_API_BASE,
        )

    # 核心方法
    def embed(self, texts: List[str]) -> List[List[float]]:
        """把文本列表转成向量列表"""
        if self.mode == "local":
            # 记得开归一化
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        elif self.mode == "api":
        #     response = self.client.embeddings.create(
        #         model=Config.EMBEDDING_API_MODEL,
        #         input=texts
        #     )
        #     return [data.embedding for data in response.data]
            raise ValueError(f"API 模式暂未实现")
        else:
            raise ValueError(f"不支持的 embedding 模式：{self.mode}")

    # 资源管理
    def unload(self):
        """释放模型内存（索引完成后调用，日常聊天不驻留）"""
        if self.model is not None:
            del self.model    # 引用计数归零 → 立刻释放大部分内存（~1.3GB）
            self.model = None # 确保引用被清除
            import gc
            gc.collect()
