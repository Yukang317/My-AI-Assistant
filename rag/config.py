"""
RAG 模块配置

所有配置项都支持两种方式设置（优先级从高到低）：
  1. 环境变量
  2. 下方 DEFAULT_* 开头的默认值

使用方式：
  from rag.config import Config
  print(Config.MILVUS_HOST)        # → "127.0.0.1"
  print(Config.EMBEDDING_MODE)     # → "local"
"""

import os
from dotenv import load_dotenv

load_dotenv()  # 加载 personal_assistant/.env 中的变量


class Config:
    """RAG 系统配置（所有值从环境变量读取，带默认值）"""

    # ── 1. LLM 配置（已有，从 .env 读） ──────────────────────────────
    LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    LLM_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    LLM_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))   # RAG 场景温度低一点
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))

    # ── 2. Embedding 配置 ────────────────────────────────────────
    # 模式: "local"（BGE 本地）/ "api"（调用远程 API）
    EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local")

    # 本地模式 — BGE 模型
    # bge-large-zh-v1.5: 1024维, ~1.3GB, 中文召回率 81.5%
    # bge-small-zh-v1.5: 512维, ~100MB, 轻量备选
    EMBEDDING_LOCAL_MODEL = os.getenv(
        "EMBEDDING_LOCAL_MODEL",
        "BAAI/bge-large-zh-v1.5"
    )
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

    # API 模式（备选，暂不用）
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "")
    EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "text-embedding-3-small")

    # ── 3. Milvus 向量数据库配置 ──────────────────────────────────
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_USER = os.getenv("MILVUS_USER", "root")
    MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD", "admin123")

    # 两个 Collection：父集合存完整上下文，子集合做精准检索
    MILVUS_PARENT_COLLECTION = os.getenv("MILVUS_PARENT_COLLECTION", "parent_chunks")
    MILVUS_CHILD_COLLECTION = os.getenv("MILVUS_CHILD_COLLECTION", "child_chunks")

    # 搜索参数
    MILVUS_SEARCH_NPROBE = int(os.getenv("MILVUS_SEARCH_NPROBE", "64"))
    MILVUS_INDEX_NLIST = int(os.getenv("MILVUS_INDEX_NLIST", "1024"))

    # ── 4. PostgreSQL 配置 ───────────────────────────────────────
    # 复用 ECS 已有的 PostgreSQL 17（端口 15432）
    PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
    PG_PORT = os.getenv("PG_PORT", "15432")
    PG_DATABASE = os.getenv("PG_DATABASE", "personal_assistant")
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")

    # ── 5. 文档分块配置 ──────────────────────────────────────────
    # 父子分块：父块保留完整上下文，子块做精准检索
    PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2048"))
    PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "128"))
    CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "512"))
    CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "64"))

    # ── 6. 检索配置 ──────────────────────────────────────────────
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))       # 向量检索取多少个
    RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))              # 重排后保留几个
    USE_RERANK = os.getenv("USE_RERANK", "false").lower() == "true" # 默认关闭重排

    # ── 7. MinIO 对象存储配置 ───────────────────────────────────
    # 存所有原始文件：PDF、Word、Markdown、图片、视频等
    MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000")
    MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_BUCKET = os.getenv("MINIO_BUCKET", "personal-assistant")
    MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))  # 默认 50MB

    # 允许上传的文件类型
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

    # ── 8. RAG Prompt 模板 ───────────────────────────────────────
    RAG_PROMPT_TEMPLATE = """你是一个个人 AI 助理。请根据用户的知识库内容回答问题。

知识库内容:
{context}

用户问题: {question}

回答要求：
1. 严格基于知识库内容回答，不要编造信息
2. 如果知识库中有答案，请准确完整地回答，并在末尾标注来源文档
3. 如果知识库中信息不完整，请说明现有信息，建议用户查看原始文档
4. 如果知识库中没有相关信息，请诚实说明"我的知识库中没有找到相关信息"
5. 使用 Markdown 格式组织回答，让内容清晰易读"""


# ── 启动时打印关键配置（方便调试） ─────────────────────────────────
if __name__ == "__main__":
    print("=== RAG 配置 ===")
    print(f"Embedding 模式: {Config.EMBEDDING_MODE}")
    print(f"Embedding 模型: {Config.EMBEDDING_LOCAL_MODEL}")
    print(f"向量维度: {Config.EMBEDDING_DIM}")
    print(f"Milvus: {Config.MILVUS_HOST}:{Config.MILVUS_PORT}")
    print(f"PostgreSQL: {Config.PG_HOST}:{Config.PG_PORT}/{Config.PG_DATABASE}")
    print(f"父块大小: {Config.PARENT_CHUNK_SIZE}, 重叠: {Config.PARENT_CHUNK_OVERLAP}")
    print(f"子块大小: {Config.CHILD_CHUNK_SIZE}, 重叠: {Config.CHILD_CHUNK_OVERLAP}")
    print(f"检索 Top-K: {Config.RETRIEVAL_TOP_K}, 重排: {'开' if Config.USE_RERANK else '关'}")
    print(f"MinIO: {Config.MINIO_ENDPOINT}, Bucket: {Config.MINIO_BUCKET}")
    print(f"最大文件: {Config.MAX_FILE_SIZE // 1024 // 1024}MB")
    print(f"LLM: {Config.LLM_MODEL}")
