"""
个人 AI 助理 — 后端服务
运行：uv run step3_api.py
"""

import os
import sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
    question: str


class ChatResponse(BaseModel):
    reply: str


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
    conn.commit()
    conn.close()


def load_messages(session_id=SESSION_ID):
    """读取指定会话的全部历史消息，按时间排序"""
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


# ── API 端点 ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """POST /chat：接收问题 → 加载历史 → 调 DeepSeek → 保存结果 → 返回"""

    # 加载历史（首次会话自动补 system prompt）
    messages = load_messages()
    if not messages:
        messages.append({
            "role": "system",
            "content": "你是一个乐于助人的个人助理，用中文回答所有问题。",
        })
        save_message(SESSION_ID, "system", messages[0]["content"])

    # 用户消息：写入内存 + 写入数据库
    messages.append({"role": "user", "content": req.question})
    save_message(SESSION_ID, "user", req.question)

    # 调用 DeepSeek
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    reply_content = response.choices[0].message.content

    # 助手回复：写入内存 + 写入数据库
    messages.append({"role": "assistant", "content": reply_content})
    save_message(SESSION_ID, "assistant", reply_content)

    return ChatResponse(reply=reply_content)


@app.get("/", response_class=HTMLResponse)
def index():
    """GET /：返回聊天前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ── 启动入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    uvicorn.run(app, host="0.0.0.0", port=1000)
