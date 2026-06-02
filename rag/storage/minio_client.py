"""
MinIO 客户端封装 — 存/取/删原始文件
"""
from minio import Minio
from rag.config import Config
from io import BytesIO


class MinioClient:
    """MinIO S3 兼容对象存储客户端

    职责：管理原始文件的存储和读取
    对应 MinIO bucket：personal-assistant（Config.MINIO_BUCKET）
    """
    def __init__(self):
        self.client = Minio(
            endpoint=Config.MINIO_ENDPOINT,       # MinIO 服务器地址
            access_key=Config.MINIO_ACCESS_KEY,   # MinIO 访问密钥
            secret_key=Config.MINIO_SECRET_KEY,   # MinIO 秘密密钥
            secure=Config.MINIO_SECURE,           # 是否使用 HTTPS
        )
        self.bucket = Config.MINIO_BUCKET
        self._ensure_bucket()

    def _ensure_bucket(self):
        """确保 bucket 存在，不存在则创建"""
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, file_data: bytes, object_key: str, content_type: str) -> str:
        """上传文件到 MinIO

        参数：
            file_data:    原始文件字节（parser 解析完后的二进制内容）
            object_key:   MinIO 中的对象路径，如 "docs/2026/report.pdf"
            content_type: MIME 类型，如 "application/pdf"

        返回：
            object_key（调用方需要用这个 key 存到 PostgreSQL）

        步骤：
            1. 把 file_data 包装成 io.BytesIO
            2. 调用 client.put_object()
            3. 返回 object_key
        """
        self.client.put_object(
            bucket_name=self.bucket,
            object_name = object_key,
            data = BytesIO(file_data),
            length = len(file_data),
            content_type = content_type,
        )
        return object_key

    def download(self, object_key: str) -> bytes:
        """从 MinIO 下载文件

        参数：
            object_key:   MinIO 中的对象路径，如 "docs/2026/report.pdf"

        返回：
            文件字节
        """
        try:
            response = self.client.get_object(
                self.bucket,
                object_key,
            )
            return response.read()  # 读数据
        finally:
            response.close()        # 不管是否报错，最后都关连接
            response.release_conn() # 归还连接池（不调用会导致连接泄漏）

    def delete(self, object_key: str):
        """删除文件（文件不存在时静默忽略）"""
        self.client.remove_object(self.bucket, object_key)

    def exists(self, object_key: str) -> bool:
        """判断文件是否存在"""
        try:
            self.client.stat_object(self.bucket, object_key)
            return True
        except Exception:
            return False
        




























