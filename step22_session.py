"""
个人 AI 助理 — 后端服务
"""

import uuid
import datetime
import json         # SSE 事件格式用 json.dumps()
import asyncio      # SSE 流式输出用 asyncio.sleep()


import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── 环境变量 & OpenAI 客户端 ─────────────────────────────────────────────

load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = "deepseek-chat"
client = OpenAI(api_key=api_key, base_url=base_url)

# ── FastAPI 应用 & 中间件 ────────────────────────────────────────────────

app = FastAPI(title="个人助理 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 数据模型 ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = False   # 是否启用流式输出（默认关闭，保持向后兼容）


class ChatResponse(BaseModel):
    created_at: str
    reply: str

class SessionInfo(BaseModel):
    session_id: str         # 会话 ID
    title: str             # 会话标题
    message_count: int     # 消息数量
    updated_at: str        # 最新一条消息的时间

class MessageInfo(BaseModel):
    role: str           # system / user / assistant
    content: str        # 消息内容
    created_at: str     # 创建时间

class RenameRequest(BaseModel):
    title: str


# ── SQLite 数据库操作 ────────────────────────────────────────────────────

DB_PATH = "chat.db"
SESSION_ID = "default"  # 单会话模式，后续升级多会话


def init_db():
    """建表（表不存在时才创建）"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            custom_title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def load_messages(session_id=SESSION_ID):
    """读取指定会话的全部历史消息"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


def save_message(session_id, role, content):
    """写入一条消息"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()
    conn.close()

async def generate_stream_response(session_id, messages: list[dict], model_name: str):
    """
    流式调用 DeepSeek，逐块产出 SSE 事件字符串。

    工作流程：
      1. 调用 OpenAI API（stream=True），获得一个流式迭代器
      2. 遍历每个 chunk，提取 delta.content（增量文本）
      3. 累加完整回复内容（流结束后写入数据库）
      4. 每收到一个 chunk 就 yield 一条 SSE 事件给前端
      5. 流结束后：把完整回复写入 SQLite，再 yield 结束信号

    SSE 事件格式：data: {"content":"...", "finished":false, "created_at":"..."}\n\n

    Args:
        session_id: 当前会话 ID，用于写入数据库
        messages: 完整的消息历史（含 system prompt + 用户问题）
        model_name: 模型名称（"deepseek-chat"）

    Yields:
        str: 格式为 "data: {json}\n\n" 的 SSE 事件字符串
    """
    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        accumulated = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                # 检查当前 chunk 是否包含有效的文本内容
                # 有些 chunk 是空的（如流开始的元数据），需要跳过
                content = chunk.choices[0].delta.content
                accumulated += content
                event ={
                    "content": content,         # 本次增量文本
                    "finished": False,          # 尚未结束
                    "created_at": datetime.datetime.now().isoformat(),
                }
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)
        
        if accumulated:
            save_message(session_id, "assistant", accumulated)
        
        # end_event 放外面，不管有没有内容都通知前端"流结束了"
        end_event = {
            "content": "",
            "finished":True,
            "created_at": datetime.datetime.now().isoformat(),
        }
        yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_event = {
            "error": str(e),
            "finished": True,
            "created_at": datetime.datetime.now().isoformat(),
        }
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"


def list_sessions():
    """列出所有会话"""
    conn = sqlite3.connect(DB_PATH)
    # 查询的 m 是messages表，所以空会话不会被查询
    rows = conn.execute("""
        SELECT
            m.session_id,
            COALESCE(
                s.custom_title,
                (SELECT content FROM messages WHERE session_id = m.session_id AND role = 'user' 
                ORDER BY id LIMIT 1) 
            )as title,
            COUNT(*) as msg_count,
            MAX(m.created_at) as updated_at
        FROM messages m
        LEFT JOIN sessions s ON m.session_id = s.session_id
        GROUP BY m.session_id
        ORDER BY updated_at DESC
    """).fetchall()
# 示例（LEFT JOIN sessions 后，custom_title 优先）：
#   rows = [
#       ("abc123", "自定义标题A",     3, "2026-06-01 10:02"),  # sessions.custom_title 覆盖了首条消息
#       ("def456", "今天天气如何",     2, "2026-05-27 11:01"),  # sessions.custom_title 为 NULL，用首条消息
#   ]

    conn.close()
    # 每行 tuple 转成SessionInfo对象
    return [
        SessionInfo(session_id=row[0], title=row[1] or "新会话", message_count=row[2], updated_at=row[3])
        for row in rows
    ]

    

def delete_session(session_id):
    """删除指定会话"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM messages WHERE session_id = ?",
        (session_id,),
    )
    conn.execute(
        "DELETE FROM sessions WHERE session_id = ?",
        (session_id,),
    )
    conn.commit()
    conn.close()

def load_messages_with_time(session_id):
    """读取指定会话的全部历史消息，按时间排序"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1], "created_at": row[2]} for row in rows]

# ── API 端点 ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """GET /：返回聊天前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    POST /chat：接收问题 → 加载历史 → 调 DeepSeek → 保存结果 → 返回

    支持两种模式（由请求体中的 stream 字段控制）：
      - 非流式（stream=false，默认）：等待完整回复后返回 JSON（ChatResponse）
      - 流式（stream=true）：通过 SSE 逐块返回文本（StreamingResponse）
    """

    # 加载历史（首次会话自动补 system prompt）
    messages = load_messages(req.session_id)
    if not messages:
        messages.append({
            "role": "system",
            "content": "你是一个乐于助人的个人助理，用中文回答所有问题。",
        })
        save_message(req.session_id, "system", messages[0]["content"])

    # 用户消息：写入内存 + 写入数据库
    messages.append({"role": "user", "content": req.question})
    save_message(req.session_id, "user", req.question)

    if req.stream:
        # ========== 流式路径：SSE ==========
        return StreamingResponse(
            generate_stream_response(req.session_id, messages, model),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control":"no-cache",
                "Connection":"keep-alive",
            },
        )
    else:
        # ========== 非流式路径：JSON（完全保持原有逻辑） ==========
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        reply_content = response.choices[0].message.content

        # 助手回复：写入内存 + 写入数据库
        messages.append({"role": "assistant", "content": reply_content})
        save_message(req.session_id, "assistant", reply_content)

        return ChatResponse(
            created_at=datetime.datetime.now().isoformat(), # 2026-05-27 14:30:00.123456 转为 "2026-05-27T14:30:00.123456"
            reply=reply_content)

@app.post("/sessions")
def create_session():
    """POST /sessions：创建新会话，返回session_id"""
    new_id = uuid.uuid4().hex # 32位随机字符串
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id) VALUES (?)",
        (new_id,),
    )
    conn.commit()
    conn.close()
    return {"session_id": new_id}

@app.get("/sessions")
def get_sessions():
    """GET /sessions：列出所有会话（侧边栏用）"""
    return list_sessions() # 返回一个list，fastapi会自动转为json

# 路径参数 {session_id} 会自动从 URL 里提取值传给函数参数，这是 FastAPI 的内置特性。
@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """DELETE /sessions/{session_id}：删除指定会话"""
    delete_session(session_id)
    return {"ok": True}

@app.put("/sessions/{session_id}/rename")
def rename_session(session_id: str, req: RenameRequest):
    """
    PUT /sessions/{session_id}/rename：重命名会话

    请求体：{"title": "新标题"}
    成功返回：{"ok": true, "title": "新标题"}
    失败返回：{"ok": false, "error": "标题不能为空"}
    """
    title = req.title.strip()
    if not title:
        return {"ok": False, "error": "标题不能为空"}
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (session_id, custom_title) VALUES (?, ?)"
        "ON CONFLICT(session_id) DO UPDATE SET custom_title = ?",
        (session_id, title, title),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "title": title}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """GET /sessions/{session_id}/messages：获取指定会话的所有消息"""
    messages_list = load_messages_with_time(session_id)
    return messages_list

# ── 启动入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=1000)
