"""
个人 AI 助理 — 后端服务
"""

import uuid
import datetime
import json          # SSE 事件格式用 json.dumps()
import asyncio       # 流式生成器中的 await asyncio.sleep()

import os
import db  # 数据库抽象层（PostgreSQL），替代原来的 sqlite3
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse  # StreamingResponse 用于 SSE 流式输出
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# RAG 模块导入
from rag.config import Config
from rag.indexer import DocumentIndexer
from rag.rag_service import RagService
from rag.embedding import EmbeddingService
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.rrf_fusion import RRFFusion
from rag.storage.vector_store import MilvusVectorStore
from fastapi import UploadFile, File, Form  # 文件上传

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

# 托管静态文件（JS/CSS 等，避免 CDN 被墙问题）
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── 数据模型 ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str
    stream: bool = False    # 是否启用流式输出（默认关闭，保持向后兼容）
    use_rag: bool = False   # 是否使用知识库检索


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

class DocumentInfo(BaseModel):
    """文档列表项的响应模型"""
    id: int
    filename: str
    file_type: str
    file_size: int
    object_key: str
    file_md5: str
    parent_count: int
    child_count: int
    created_at: str

class RagChatResponse(BaseModel):
    """RAG 模式下的聊天响应模型（比普通 ChatResponse 多了 sources）"""
    created_at: str
    reply: str
    sources: list[dict] = []    # 参考文档列表
    token_usage: dict = {}      # LLM 的 token 使用情况
    latency_ms: int = 0         # 检索 + 生成的总延迟


# —— 数据库操作（全部委托给 db.py，底层 PostgreSQL）


async def generate_stream_response(session_id: str, messages: list[dict], model_name: str):
    """
    流式调用 DeepSeek，逐块产出 SSE 事件字符串。

    工作流程：
      1. 调用 OpenAI API（stream=True），获得一个流式迭代器
      2. 遍历每个 chunk，提取 delta.content（增量文本）
      3. 累加完整回复内容（流结束后写入数据库）
      4. 每收到一个 chunk 就 yield 一条 SSE 事件给前端
      5. 流结束后：把完整回复写入 PostgreSQL，再 yield 结束信号

    SSE 事件格式：data: {"content":"...", "finished":false, "created_at":"..."}\n\n

    Args:
        session_id: 当前会话 ID，用于写入数据库
        messages: 完整的消息历史（含 system prompt + 用户问题）
        model_name: 模型名称（"deepseek-chat"）

    Yields:
        str: 格式为 "data: {json}\n\n" 的 SSE 事件字符串
    """
    try:
        # ── 调用 DeepSeek 流式 API ──
        # stream=True 让 OpenAI SDK 返回一个迭代器，而不是等完整响应
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            stream=True,
        )

        # ── 逐块读取并转发给前端 ──
        accumulated = ""            # 累积完整回复文本，用于最终写入数据库
        for chunk in stream:
            # 检查当前 chunk 是否包含有效的文本内容
            # 有些 chunk 是空的（如流开始的元数据），需要跳过
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                accumulated += content

                # 构建 SSE 事件数据
                event = {
                    "content": content,           # 本次增量文本（可能只有1-2个字）
                    "finished": False,            # 尚未结束
                    "created_at": datetime.datetime.now().isoformat(),
                }
                # SSE 协议格式：data: {json}\n\n
                # ensure_ascii=False 保留中文原文不转义
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                # 让出控制权，避免阻塞事件循环（好习惯，虽然单用户影响不大）
                await asyncio.sleep(0.01)

        # ── 流结束：保存完整回复到数据库 ──
        if accumulated:
            db.save_message(session_id, "assistant", accumulated)

        # ── 发送结束信号：通知前端流已完成 ──
        final = {
            "content": "",                       # 结束事件无内容
            "finished": True,                    # 标记流结束
            "created_at": datetime.datetime.now().isoformat(),
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"

    except Exception as e:
        # ── 错误处理：通知前端流已中断 ──
        error_event = {
            "error": str(e),                     # 错误信息
            "finished": True,                    # 标记流结束（异常终止）
            "created_at": datetime.datetime.now().isoformat(),
        }
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"


async def generate_rag_stream_response(session_id: str, question: str):
    """
    RAG 流式查询的 SSE 生成器。

    和 generate_stream_response 的区别：
      - 检索在 LLM 之前完成，先推送 sources 事件（引用了哪些文档）
      - LLM 调用走 RagService.query_stream() 而非直接调 DeepSeek
      - SSE 格式增加 type 字段：sources / delta / complete / error

    Args:
        session_id: 当前会话 ID，用于最终写入数据库
        question: 用户问题（已保存到 DB，这里只用于检索 + 生成）

    Yields:
        str: SSE 事件字符串
    """
    try:
        rag_service = get_rag_service()
        accumulated = ""

        # query_stream 返回 AsyncGenerator，每次 yield 一个 dict：
        #   {"type": "sources",  "documents": [...]}     → 参考文档列表
        #   {"type": "delta",    "content": "你好"}       → LLM 逐 token
        #   {"type": "complete", "token_usage": {...}}    → 结束信号
        #   {"type": "error",    "content": "..."}        → 异常
        async for event in rag_service.query_stream(question):
            event_type = event.get("type")

            if event_type == "sources":
                sse = {
                    "type": "sources",
                    "sources": event.get("documents", []),
                    "finished": False,
                }
                yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"

            elif event_type == "delta":
                content = event.get("content", "")
                accumulated += content
                sse = {
                    "type": "delta",
                    "content": content,
                    "finished": False,
                }
                yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.01)

            elif event_type == "complete":
                if accumulated:
                    db.save_message(session_id, "assistant", accumulated)
                sse = {
                    "type": "complete",
                    "content": "",
                    "finished": True,
                    "token_usage": event.get("token_usage", {}),
                }
                yield f"data: {json.dumps(sse, ensure_ascii=False)}\n\n"

            elif event_type == "error":
                raise Exception(event.get("content", "RAG 查询未知错误"))

    except Exception as e:
        # 把底层技术异常映射为用户能看懂的中文提示
        error_msg = str(e)
        if "Milvus" in error_msg or "Connection" in error_msg:
            user_error = "知识库服务暂不可用，请稍后重试。如果问题继续，请检查 Milvus 服务是否正常运行。"
        elif "索引尚未构建" in error_msg:
            user_error = "知识库中暂无文档，请先上传文档后再开启知识库提问。"
        else:
            user_error = f"知识库查询出错：{error_msg}"

        error_event = {
            "error": user_error,
            "finished": True,
        }
        yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
        
        
        


# —— RAG 服务懒加载 ———————————————————————————————————————————————————————

_rag_service: RagService | None = None      # 知识检索：接收查询、混合检索、重排、LLM生成。加载BM25索引、BGE、重排模型等
_indexer: DocumentIndexer | None = None     # 文档摄入：解析、分块、向量化、存入；rag离线。连接Milvus、MinIO、加载嵌入模型

def get_rag_service() -> RagService:
    """懒加载 Ragservice（首次调用时从Milvus 构建 BM25 + 加载 BGE 模型"""
    global _rag_service
    if _rag_service is not None:
        return _rag_service

    print("[RAG] 正在初始化 RAG 服务...")
    vs = MilvusVectorStore()        # 向量存储实例：将文档向量数据存入Milvus
    emb = EmbeddingService()        # 嵌入服务实例

    # 从 Milvus 读全量 child_chunks 构建 BM25。
    # - 所有文档的文本内容都已经存在 Milvus 里了（在 MILVUS_CHILD_COLLECTION 这个集合中）
    # - BM25 需要什么？需要所有文档的文本。既然文本已经在 Milvus 里了，那就直接从 Milvus 读出来，不用再单独存一份。
    bm25 = BM25Index(language="zh")
    try:
        # 从Milvus的子块集合中查出所有chunk（文本内容）
        all_chunks = vs.client.query(
            collection_name = Config.MILVUS_CHILD_COLLECTION,
            filter=f'embedding_model == "{Config.get_embedding_model_name()}"',
            output_fields=["child_id", "content"],
            limit=2000,
        )
        # all_chunks = [
        #     {"child_id": "chunk_001", "content": "DeepSeek API 是一个..."},
        #     {"child_id": "chunk_002", "content": "向量检索的原理是..."},
        #     # ... 最多 2000 条
        # ]

        # 建立关键词索引，搜“关键词”得到包含“关键词”的chunk
        if all_chunks:
            bm25.build_index(
                [c["content"] for c in all_chunks],
                [c["child_id"] for c in all_chunks],
            )
            print(f"[RAG] BM25 索引已从 Milvus 构建完成，包含 {len(all_chunks)} 个文档")
        else:
            print("[RAG] Milvus 中暂无数据，无法构建 BM25 索引")
        
    except Exception as e:
        print(f"[RAG] 从 Milvus 构建 BM25 索引时出错：{e}")
    
    retriever = VectorRetriever(emb, vs)
    rrf = RRFFusion(k=60)
    _rag_service = RagService(
        bm25_index=bm25,             # 有状态索引，建立好后保存在内存中
        vector_retriever=retriever,  # 无状态索引，milvus已经存好内容了，只需要搜索
        rrf_fusion=rrf,
    )
    print("[RAG] RAG 服务初始化完成")
    return _rag_service

# 上传文档时调用
def get_indexer() -> DocumentIndexer:
    """懒加载 Indexer（首次调用时连接 Milvus、MinIO、加载嵌入模型）"""
    global _indexer
    if _indexer is None:
        _indexer = DocumentIndexer()    # 内部初始化了parser、embedder、vector_store、minio
        print("[RAG] 索引器初始化完成")
    return _indexer

def update_bm25_after_uploads(chunks_text: list[str], chunk_ids: list[str]) -> None:
    """文档上传后增量更新 BM25 索引"""
    if _rag_service is None:
        return print("[RAG] RAG 服务未初始化，无法更新 BM25 索引")
    # 如果有上传新文档，全量重建BM250kapi模型
    if chunks_text:
        _rag_service.bm25_index.add_chunks(
            chunks_text,
            chunk_ids,
        )




# ── API 端点 ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """GET /：返回聊天前端页面"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    POST /chat：接收问题 → 加载历史 → 调 LLM → 保存结果 → 返回

    四种模式（由 stream + use_rag 组合控制）：
      - stream=false, use_rag=false：直调 DeepSeek → JSON
      - stream=true,  use_rag=false：直调 DeepSeek → SSE 流式
      - stream=false, use_rag=true ：RAG 检索 → DeepSeek → JSON（含 sources）
      - stream=true,  use_rag=true ：RAG 检索 → DeepSeek → SSE 流式（含 sources）
    """

    # ── 加载历史 & 首次自动补 system prompt ──
    messages = db.load_messages(req.session_id)
    if not messages:
        messages.append({
            "role": "system",
            "content": "你是一个乐于助人的个人助理，用中文回答所有问题。",
        })
        db.save_message(req.session_id, "system", messages[0]["content"])

    # ── 用户消息：写入内存 + 写入数据库 ──
    messages.append({"role": "user", "content": req.question})
    db.save_message(req.session_id, "user", req.question)

    # ═══════════════════════════════════════════════════════════
    # 🆕 RAG 分支：use_rag=True 时走检索增强生成
    # ═══════════════════════════════════════════════════════════
    if req.use_rag:
        if req.stream:
            # RAG 流式：SSE（先推 sources，再逐块推文本）
            return StreamingResponse(
                generate_rag_stream_response(req.session_id, req.question),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # RAG 非流式：检索 → 重排 → LLM 生成 → 返回 JSON
            rag_service = get_rag_service()
            result = rag_service.query(req.question)

            # 助手回复写入数据库
            db.save_message(req.session_id, "assistant", result["answer"])

            return RagChatResponse(
                created_at=datetime.datetime.now().isoformat(),
                reply=result["answer"],
                sources=result.get("sources", []),
                token_usage=result.get("token_usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

    # ═══════════════════════════════════════════════════════════
    # 普通聊天分支（use_rag=False，完全保持原有逻辑）
    # ═══════════════════════════════════════════════════════════
    if req.stream:
        # ========== 流式路径：SSE ==========
        return StreamingResponse(
            generate_stream_response(req.session_id, messages, model),
            media_type="text/event-stream",       # SSE 的标准 MIME 类型
            headers={
                "Cache-Control": "no-cache",      # 禁止浏览器/代理缓存 SSE 流
                "Connection": "keep-alive",       # 保持 TCP 连接不关闭
            },
        )
    else:
        # ========== 非流式路径：JSON ==========
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        reply_content = response.choices[0].message.content

        # 助手回复：写入内存 + 写入数据库
        messages.append({"role": "assistant", "content": reply_content})
        db.save_message(req.session_id, "assistant", reply_content)

        return ChatResponse(
            created_at=datetime.datetime.now().isoformat(),
            reply=reply_content,
        )

@app.post("/sessions")
def create_session():
    """POST /sessions：创建新会话，返回session_id"""
    new_id = uuid.uuid4().hex  # 32位随机字符串
    db.create_session(new_id)  # 显式创建 sessions 记录
    return {"session_id": new_id}

@app.get("/sessions")
def get_sessions():
    """GET /sessions：列出所有会话（侧边栏用）"""
    return db.list_sessions()  # 返回一个list，fastapi会自动转为json

# 路径参数 {session_id} 会自动从 URL 里提取值传给函数参数，这是 FastAPI 的内置特性。
@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """DELETE /sessions/{session_id}：删除指定会话"""
    db.delete_session(session_id)
    return {"ok": True}

@app.put("/sessions/{session_id}/rename")
def rename_session(session_id: str, payload: dict):
    """PUT /sessions/{session_id}/rename：重命名会话"""
    title = payload.get("title", "").strip()
    if not title:
        return {"ok": False, "error": "标题不能为空"}
    ok = db.rename_session(session_id, title)
    return {"ok": ok}

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """GET /sessions/{session_id}/messages：获取指定会话的所有消息"""
    messages_list = db.load_messages_with_time(session_id)
    return messages_list


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    POST /api/documents/upload：上传文档，完成 解析→分块→向量化→MinIO+Milvus 双存储

    请求格式：multipart/form-data，字段名 file
    响应：DocumentInfo（含 object_key、file_md5、父子块数量等）
    """
    # 1. 读取上传文件的全部字节
    file_data = await file.read()

    # 2. 调用索引器：解析 → 分块 → 向量化 → MinIO + Milvus 双写
    indexer = get_indexer()
    result = indexer.index_document(file_data, file.filename)

    # 跳过已存在的文件（内容相同的文件不做重复处理）
    if result["status"] == "skipped":
        return result

    # 3. 保存文档元数据到 SQLite（前端文档列表用）
    doc_id = db.save_document(
        filename=file.filename,
        file_type=result["file_type"],
        file_size=len(file_data),
        object_key=result["object_key"],
        file_md5=result["file_md5"],
        parent_count=result["parent_count"],
        child_count=result["child_count"],
    )

    # 4. 如果 RAG 服务已初始化，增量更新 BM25 索引
    #    复用 _rag_service 内部的 vector_store，不重复 new MilvusVectorStore()
    if _rag_service is not None:
        try:
            vs = _rag_service.vector_retriever.vector_store
            child_chunks = vs.client.query(
                collection_name=Config.MILVUS_CHILD_COLLECTION,
                filter=f'doc_path_name == "{result["object_key"]}"',
                output_fields=["child_id", "content"],
                limit=10000,
            )
            if child_chunks:
                update_bm25_after_uploads(
                    [c["content"] for c in child_chunks],
                    [c["child_id"] for c in child_chunks],
                )
        except Exception as e:
            print(f"[RAG] 上传后 BM25 更新失败（不影响上传本身）: {e}")

    # 5. 返回文档信息（含数据库自增 ID）
    return {
        "id": doc_id,
        "filename": file.filename,
        "file_type": result["file_type"],
        "file_size": len(file_data),
        "object_key": result["object_key"],
        "file_md5": result["file_md5"],
        "parent_count": result["parent_count"],
        "child_count": result["child_count"],
        "status": result["status"],
    }

@app.get("/api/documents")
def get_documents():
    """
    GET /api/documents: 列出所有已上传的文档，按创建时间倒序

    返回：List[DocumentInfo]
    """
    return db.list_documents()

@app.delete("/api/documents/{object_key:path}")
async def delete_document(object_key: str):
    """
    DELETE /api/documents/{object_key}：删除指定文档（三存储清理）

    :path 转换器让 object_key 里的 / 不被 FastAPI 当成路由分隔符截断。

    删除顺序（重要！）：① BM25 先移除 → ② MinIO+Milvus 删 → ③ SQLite 删
    为什么 BM25 要在 Milvus 删除之前？因为 remove_by_doc_ids 不依赖 Milvus，
    但如果先删 Milvus 再删 BM25，中间出错了 BM25 里还留着脏数据。
    """
    # ① 如果 RAG 服务已初始化，先从 BM25 内存索引中移除该文档
    #    复用 _rag_service 内部的 vector_store，不重复 new
    if _rag_service is not None:
        try:
            vs = _rag_service.vector_retriever.vector_store
            # 在 Milvus 删除之前查出该文档的所有 child_id
            child_chunks = vs.client.query(
                collection_name=Config.MILVUS_CHILD_COLLECTION,
                filter=f'doc_path_name == "{object_key}"',
                output_fields=["child_id"],
                limit=10000,
            )
            child_ids_to_remove = [c["child_id"] for c in child_chunks]
            if child_ids_to_remove:
                removed = _rag_service.bm25_index.remove_by_doc_ids(child_ids_to_remove)
                print(f"[RAG] BM25 已移除 {removed} 个 chunk（文档: {object_key}）")
        except Exception as e:
            print(f"[RAG] BM25 移除失败（不影响删除本身）: {e}")

    # ② 从 MinIO + Milvus 删向量和原始文件
    indexer = get_indexer()
    delete_result = indexer.delete_document(object_key)

    # ③ 从 SQLite 删除元数据
    db_deleted = db.delete_document_by_key(object_key)

    return {
        "ok": True,
        "object_key": object_key,
        "parent_deleted": delete_result["parent_deleted"],
        "child_deleted": delete_result["child_deleted"],
        "db_deleted": db_deleted,
    }







# ── 启动入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db.init_db()
    print("[启动] SQLite 数据库已初始化（messages + documents）")
    print("[启动] RAG 服务采用懒加载，首次 RAG 请求时初始化（连接 Milvus + 加载 BGE + 构建 BM25）")
    uvicorn.run(app, host="0.0.0.0", port=1000)
