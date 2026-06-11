"""
db.py — 数据库抽象层（PostgreSQL）

封装所有数据库操作，替代之前的 sqlite3 直接调用。
所有函数签名与旧版 SQLite 版保持一致，方便 step23_rag.py 无缝切换。

连接参数从 rag.config.Config 读取，支持通过环境变量覆盖。

使用方式：
    from db import init_db, load_messages, save_message, ...
"""

import psycopg2
from rag.config import Config


def _get_conn():
    """
    获取 PostgreSQL 连接。

    每次调用创建新连接，用完即关（当前单用户场景足够）。
    后续如果需要连接池，只需修改此函数即可。
    """
    return psycopg2.connect(
        host=Config.PG_HOST,
        port=Config.PG_PORT,
        dbname=Config.PG_DATABASE,
        user=Config.PG_USER,
        password=Config.PG_PASSWORD,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 建表
# ═════════════════════════════════════════════════════════════════════════════


def init_db():
    """
    创建所有业务表（表不存在时才创建）。

    PG 与 SQLite 建表差异：
      - INTEGER PRIMARY KEY AUTOINCREMENT → SERIAL PRIMARY KEY
      - TEXT / TIMESTAMP 语法相同
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # sessions 表：记录每个会话的 ID 和自定义标题
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    custom_title TEXT,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # messages 表：聊天消息记录
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          SERIAL PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 为 session_id 建索引，加速按会话查询消息
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id)
            """)

            # documents 表：文档台账（RAG 摄入元数据）
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id           SERIAL PRIMARY KEY,
                    filename     TEXT NOT NULL,
                    file_type    TEXT NOT NULL,
                    file_size    INTEGER NOT NULL,
                    object_key   TEXT NOT NULL UNIQUE,
                    file_md5     TEXT NOT NULL,
                    parent_count INTEGER DEFAULT 0,
                    child_count  INTEGER DEFAULT 0,
                    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# 会话操作
# ═════════════════════════════════════════════════════════════════════════════


def create_session(session_id: str) -> str:
    """
    创建新会话，返回 session_id。

    旧版 SQLite 中 POST /sessions 不操作数据库（等第一条消息才写入），
    现在改为显式创建 sessions 记录，标题初始为空。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id) VALUES (%s)",
                (session_id,),
            )
        conn.commit()
        return session_id
    finally:
        conn.close()


def list_sessions() -> list:
    """
    列出所有会话，按最近活动时间倒序。

    返回每个会话的 session_id、标题、消息数、最后更新时间。
    标题优先级：custom_title（用户手动设置）> 首条用户消息 > "新会话"。

    返回 list[dict]，调用方自行构建 Pydantic 模型。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    s.session_id,
                    COALESCE(
                        s.custom_title,
                        (SELECT m.content
                         FROM messages m
                         WHERE m.session_id = s.session_id AND m.role = 'user'
                         ORDER BY m.id LIMIT 1),
                        '新会话'
                    ) AS title,
                    COUNT(m.id) AS msg_count,
                    COALESCE(MAX(m.created_at), s.created_at) AS updated_at
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id, s.custom_title, s.created_at
                ORDER BY updated_at DESC
            """)
            rows = cur.fetchall()
        return [
            {
                "session_id": row[0],
                "title": row[1],
                "message_count": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]
    finally:
        conn.close()


def rename_session(session_id: str, custom_title: str) -> bool:
    """
    设置会话的自定义标题。

    返回是否成功（session_id 存在时返回 True）。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET custom_title = %s WHERE session_id = %s",
                (custom_title, session_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def delete_session(session_id: str):
    """
    删除指定会话及其所有消息。

    先删 messages（子表），再删 sessions（主表）。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# 消息操作
# ═════════════════════════════════════════════════════════════════════════════


def load_messages(session_id: str) -> list[dict]:
    """
    读取指定会话的全部历史消息（不含时间戳，供 LLM 上下文使用）。

    返回 [{"role": "user", "content": "..."}, ...]
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content FROM messages WHERE session_id = %s ORDER BY id",
                (session_id,),
            )
            rows = cur.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]
    finally:
        conn.close()


def save_message(session_id: str, role: str, content: str):
    """
    写入一条消息，同时确保 sessions 表有对应记录。

    如果 session_id 在 sessions 表中不存在（比如旧数据），自动补建。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            # 确保 session 记录存在（幂等：已有则忽略）
            cur.execute(
                "INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT (session_id) DO NOTHING",
                (session_id,),
            )
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, %s, %s)",
                (session_id, role, content),
            )
        conn.commit()
    finally:
        conn.close()


def load_messages_with_time(session_id: str) -> list[dict]:
    """
    读取指定会话的全部历史消息（含时间戳，供前端展示用）。

    返回 [{"role": "user", "content": "...", "created_at": "..."}, ...]
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = %s ORDER BY id",
                (session_id,),
            )
            rows = cur.fetchall()
        return [
            {"role": row[0], "content": row[1], "created_at": str(row[2])}
            for row in rows
        ]
    finally:
        conn.close()


# ═════════════════════════════════════════════════════════════════════════════
# 文档操作
# ═════════════════════════════════════════════════════════════════════════════


def save_document(
    filename: str,
    file_type: str,
    file_size: int,
    object_key: str,
    file_md5: str,
    parent_count: int,
    child_count: int,
) -> int:
    """
    写入一条文档记录，返回自增 ID。

    PG 中使用 RETURNING id 获取自增主键（替代 SQLite 的 cursor.lastrowid）。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (filename, file_type, file_size, object_key, file_md5, parent_count, child_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (filename, file_type, file_size, object_key, file_md5, parent_count, child_count),
            )
            doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id
    finally:
        conn.close()


def list_documents() -> list[dict]:
    """
    列出所有已索引文档，按创建时间倒序。

    返回 list[dict]，调用方自行构建 Pydantic 模型。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, filename, file_type, file_size, object_key,
                       file_md5, parent_count, child_count, created_at
                FROM documents
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "file_size": row[3],
                "object_key": row[4],
                "file_md5": row[5],
                "parent_count": row[6],
                "child_count": row[7],
                "created_at": str(row[8]),
            }
            for row in rows
        ]
    finally:
        conn.close()


def delete_document_by_key(object_key: str) -> bool:
    """
    按 object_key 删除文档记录，返回是否删除成功。
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE object_key = %s",
                (object_key,),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()
