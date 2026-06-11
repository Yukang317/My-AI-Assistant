"""
migrate_sqlite_to_pg.py — SQLite → PostgreSQL 数据迁移脚本

一次性脚本：从 chat.db 读取全部数据，写入 PostgreSQL personal_assistant 库。
源数据只读不删，迁移失败可安全重跑（先清空 PG 表即可）。

用法：
    uv run migrate_sqlite_to_pg.py
"""

import sqlite3
from db import _get_conn

SQLITE_DB = "chat.db"


def migrate():
    # 1. 连接 SQLite，读取所有数据
    sqlite = sqlite3.connect(SQLITE_DB)

    sessions = sqlite.execute("SELECT session_id, custom_title, created_at FROM sessions").fetchall()
    messages = sqlite.execute("SELECT session_id, role, content, created_at FROM messages ORDER BY id").fetchall()
    documents = sqlite.execute(
        "SELECT filename, file_type, file_size, object_key, file_md5, parent_count, child_count, created_at FROM documents"
    ).fetchall()

    sqlite.close()

    print(f"SQLite 读取完成：sessions {len(sessions)} 条, messages {len(messages)} 条, documents {len(documents)} 条")

    # 2. 连接 PG，按外键依赖顺序写入
    pg = _get_conn()
    cur = pg.cursor()

    try:
        # 先确保表存在
        from db import init_db
        init_db()

        # 2a. 写入 sessions
        for sid, title, created_at in sessions:
            cur.execute(
                "INSERT INTO sessions (session_id, custom_title, created_at) VALUES (%s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
                (sid, title, created_at),
            )
        print(f"PG sessions 写入完成：{len(sessions)} 条")

        # 2b. 写入 messages
        for session_id, role, content, created_at in messages:
            cur.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (%s, %s, %s, %s)",
                (session_id, role, content, created_at),
            )
        print(f"PG messages 写入完成：{len(messages)} 条")

        # 2c. 写入 documents
        for filename, file_type, file_size, object_key, file_md5, parent_count, child_count, created_at in documents:
            cur.execute(
                """
                INSERT INTO documents (filename, file_type, file_size, object_key, file_md5, parent_count, child_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (object_key) DO NOTHING
                """,
                (filename, file_type, file_size, object_key, file_md5, parent_count, child_count, created_at),
            )
        print(f"PG documents 写入完成：{len(documents)} 条")

        pg.commit()
        print("\n✅ 迁移成功！chat.db 数据已完整写入 PostgreSQL。")
        print("   chat.db 保留作为备份，未被删除。")

    except Exception as e:
        pg.rollback()
        print(f"\n❌ 迁移失败：{e}")
        print("   PG 数据已回滚，SQLite 源数据未受影响。可修复问题后重新运行。")
        raise
    finally:
        cur.close()
        pg.close()


if __name__ == "__main__":
    migrate()
