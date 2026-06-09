"""
rag/evaluate.py — RAG 评估模块

基于 RAGAS 框架的轻量编排层：从 Milvus 自动采样文档、生成测试集、
调用 RagService 获取回答、计算 4 个评估指标、保存 JSON 报告。

工作流程（5 步管线）：
  ① Milvus child_chunks 随机采样 → parent_id 回溯父块 → LangChain Document
  ② RAGAS TestsetGenerator 自动出题（LLM-as-judge，5 条）
  ③ 依次调 RagService.query() 获取系统回答
  ④ RAGAS evaluate() 计算：faithfulness / context_recall / context_precision / answer_relevancy
  ⑤ 保存 eval_results/ 下 JSON 报告 + 打印控制台摘要

使用方式：
  cd personal_assistant && .venv/bin/python -m rag.evaluate

参考：
  - MilDoc ragas_evaluator.py（RAGAS 集成模式）
  - RAGAS 0.4.x 文档（TestsetGenerator + evaluate API）
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from datasets import Dataset

# RAGAS（0.4.x API）
from ragas import evaluate as ragas_evaluate
from ragas.metrics.collections import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.testset import TestsetGenerator

# LangChain（用于 RAGAS 的 LLM / Embedding / Document）
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# 项目内
from rag.config import Config
from rag.embedding import EmbeddingService
from rag.rag_service import RagService
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.rrf_fusion import RRFFusion
from rag.retrieval.vector_retriever import VectorRetriever
from rag.storage.vector_store import MilvusVectorStore

logger = logging.getLogger(__name__)

# 输出目录（评估结果保存位置）
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "eval_results"

# 默认测试集大小
DEFAULT_TEST_SIZE = 5


# ═══════════════════════════════════════════════════════════════════════
# 1. RAGAS 组件初始化
# ═══════════════════════════════════════════════════════════════════════

def _build_ragas_llm(temperature: float = 0.0) -> ChatOpenAI:
    """创建 RAGAS 使用的 LLM 客户端（DeepSeek，OpenAI 兼容接口）。

    Args:
        temperature: 出题/打分的温度（0.0 = 确定性输出，保证可复现）

    Returns:
        配置好的 ChatOpenAI 实例
    """
    # TODO(human): 用 Config.LLM_API_KEY / Config.LLM_BASE_URL / Config.LLM_MODEL 初始化 ChatOpenAI，temperature 用参数值
    return ChatOpenAI(
        api_key=Config.LLM_API_KEY,
        base_url=Config.LLM_BASE_URL,
        model=Config.LLM_MODEL,
        temperature=temperature,        # 0.0
    )


def _build_ragas_embeddings() -> HuggingFaceEmbeddings:
    """创建 RAGAS 使用的 Embedding 客户端（本地 BGE 模型）。

    复用 Config 已配置的 BGE 模型路径（bge-large-zh-v1.5），
    通过 LangChain 的 HuggingFaceEmbeddings 包装，供 RAGAS TestsetGenerator 使用。

    Returns:
        配置好的 HuggingFaceEmbeddings 实例
    """
    # TODO(human): 用 Config.EMBEDDING_LOCAL_MODEL 初始化 HuggingFaceEmbeddings，device="cpu"
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_LOCAL_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True}  # L2 归一化，和Milvus IP 度量一致
    )

# ═══════════════════════════════════════════════════════════════════════
# 2. 文档加载（Milvus 采样 + 父块回溯）
# ═══════════════════════════════════════════════════════════════════════

def _load_parent_docs_from_milvus(
    n: int = DEFAULT_TEST_SIZE,
    embedding_dim: int = 1024,
) -> List[LangchainDocument]:
    """从 Milvus child_chunks 随机采样，通过 parent_id 回溯父块，返回 LangChain Document 列表。

    原理：
      随机向量（np.random.randn）与库内任意实际向量都不相似，
      在 child_chunks 中做 ANN 搜索 → 得到随机采样效果。
      从返回结果的 metadata 中提取 parent_id，再查 parent_chunks 拿到完整父块文本。

    Args:
        n: 采样数量
        embedding_dim: 向量维度（默认 1024，对应 BGE-large）

    Returns:
        LangChain Document 列表，每个 Document 的 page_content 为父块全文，
        metadata 包含 doc_name / doc_type / parent_id 等字段
    """
    # TODO(human): ① 创建 MilvusVectorStore → ② 生成随机向量 → ③ 在 child_chunks 搜索
    #              ④ 去重 parent_id → ⑤ 查 parent_chunks 拿完整文本 → ⑥ 包成 LangChain Document
    # ① 创建向量存储实例
    vs = MilvusVectorStore()

    # ② 生成随机向量（标准正态分布，与库内任意向量都不相似）
    random_vec = np.random.randn(embedding_dim).tolist()

    # ③ 搜索 child_chunks——search() 内部会自动回溯父块
    results = vs.search(query_vector=random_vec, top_k=n)

    # ④ 去重 parent_id + 包成 LangChain Document
    documents = []
    seen = set()
    for r in results:
        pid = r.get("parent_id", "")
        if pid in seen:
            continue
        seen.add(pid)
        documents.append(LangchainDocument(
            page_content=r.get("parent_content", r.get("child_content", "")),
            metadata={
                "doc_name": r.get("doc_name", "未知"),
                "doc_type": r.get("doc_type", ""),
                "parent_id": pid,
            },
        ))
    return documents


# ═══════════════════════════════════════════════════════════════════════
# 3. 测试集生成
# ═══════════════════════════════════════════════════════════════════════

def generate_testset(
    documents: List[LangchainDocument],
    test_size: int = DEFAULT_TEST_SIZE,
    output_dir: Optional[Path] = None,
) -> Dataset:
    """用 RAGAS TestsetGenerator 从文档自动生成 {问题, 标准答案, 上下文} 三元组。

    Args:
        documents: LangChain Document 列表（来自 _load_parent_docs_from_milvus）
        test_size: 生成测试样本数量
        output_dir: 保存目录（None 则默认 OUTPUT_DIR）

    Returns:
        RagasDataset，包含 question / ground_truth / contexts 列
    """
    # TODO(human): ① 调 _build_ragas_llm + _build_ragas_embeddings
    #              ② TestsetGenerator.from_langchain(llm, embeddings)
    #              ③ generator.generate_with_langchain_docs(docs, testset_size, query_distribution)
    #              ④ 保存 JSON（可选，方便后续不重复生成）
    # ① 初始化 RAGAS 组件
    llm = _build_ragas_llm(temperature=0.0)
    embeddings = _build_ragas_embeddings()

    # ② 创建 TestsetGenerator（RAGAS 0.4.x API）
    generator = TestsetGenerator.from_langchain(
        llm=llm,
        embedding_model=embeddings,       # 注意参数名是 embedding_model
    )

    # ③ 自动生成测试集
    print(f"正在从 {len(documents)} 个文档块生成 {test_size} 条测试样本...")
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=test_size,           # 注意参数名是 testset_size
        query_distribution={              # 注意参数名是 query_distribution
            "simple": 0.4,                # 简单单文档问题
            "reasoning": 0.3,             # 需要推理的问题
            "multi_context": 0.3,         # 跨多片段的问题
        },
      )
    print(f"测试集生成完成，共 {len(testset)} 条")

    # ④ 保存 JSON 备份（output_dir 为 None 则用默认 OUTPUT_DIR）
    out_dir = output_dir or OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "testset.json"
    testset.to_pandas().to_json(save_path, orient="records", force_ascii=False, indent=2)
    print(f"测试集已保存: {save_path}")

    return testset



# ═══════════════════════════════════════════════════════════════════════
# 4. RagService 初始化
# ═══════════════════════════════════════════════════════════════════════

def _init_rag_service() -> RagService:
    """初始化 RagService，注入全部依赖（Embedding → Milvus → BM25 → VectorRetriever → RRF）。

    构造顺序遵循 rag_service.py 头部示例代码的结构。

    Returns:
        完全初始化的 RagService 实例
    """
    # 1. 创建向量存储实例，内部连接到 Milvus
    vs = MilvusVectorStore()

    # 2. 从 Milvus 查询所有 child_chunks，用于构建 BM25
    embedding_model = Config.get_embedding_model_name()
    all_chunks = vs.client.query(
        collection_name=Config.MILVUS_CHILD_COLLECTION,
        filter=f'embedding_model == "{embedding_model}"',
        output_fields=["child_id", "content", "doc_name"],
        limit=2000,         # 足够大，保证覆盖当前文档量
    )

    # 3. 构建 BM25 索引（全量语料库）
    bm25 = BM25Index(language="zh")
    texts = [c["content"] for c in all_chunks]
    ids = [c["child_id"] for c in all_chunks]
    bm25.build_index(texts, ids)

    # 4. 创建向量检索器 + 融合器
    emb = EmbeddingService()    #   最后创建（加载 BGE 模型，占用内存
    retriever = VectorRetriever(emb, vs)
    rrf = RRFFusion(k=60)

    # 5. 组装 RagService
    return RagService(
        bm25_index=bm25,
        vector_retriever=retriever,
        rrf_fusion=rrf,
    )



# ═══════════════════════════════════════════════════════════════════════
# 5. 评估主流程
# ═══════════════════════════════════════════════════════════════════════

def evaluate_rag(rag_service: RagService, testset: Dataset) -> Dict[str, Any]:
    """使用 RAGAS 评估 RagService 性能。

    对 testset 中每个问题：
      1. 调用 rag_service.query(question) 获取系统回答
      2. 收集 answer（回答文本）和 contexts（检索到的文档内容）
    然后构建评估数据集，调用 ragas.evaluate() 计算 4 个指标。

    Args:
        rag_service: 已初始化的 RagService
        testset: RAGAS 生成的测试集（含 question / ground_truth 列）

    Returns:
        评估结果字典，包含各指标均值和每条样本的明细
    """
    print("开始评估 RagService 性能...")
    print(f"正在评估 {len(testset)} 个测试样本...")
    
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    for i, question in enumerate(testset["question"], 1):
        print(f" - [{i}/{len(testset)}] 查询：{question[:60]}...")
        result = rag_service.query(question)

        # 提取回答和检索到的文档文本
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        # 优先用 parent_text （传给 LLM 的完整上下文）
        ctx = [s.get("parent_text" or s.get("text", "")) for s in sources]

        questions.append(question)
        answers.append(answer)
        contexts_list.append(ctx)
        ground_truths.append(testset["ground_truth"][i-1])

    # 构建 RAGAS 评估数据集
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # 调 RAGAS 自动打分
    print("正在调用 RAGAS 自动打分...")
    llm = _build_ragas_llm(temperature=0.0)
    embeddings = _build_ragas_embeddings()
    scores = ragas_evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_recall, context_precision, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
    )

    # 处理 NaN/Inf → None（JSON 序列化兼容）。not a number / 无穷大，python中的float类型中存在，但JSON不支持
    result = scores.to_pandas().to_dict(orient="records")
    import math

    # 清理函数，把 NaN 和 inf 转成 None（JSON 可以序列化 null）
    def sanitize(obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    # cleaned = {k: sanitize(v) for k, v in result.items() if k !="question"} # 回滚
    # 更好的方式：对每行逐值清理。每一行是每个样本的评估结果，再对每一行的每个字段调用sanitize()
    cleaned_rows = []
    for row in result:
        cleaned_rows.append({k: sanitize(v) for k, v in row.items()})
    
    # 计算各指标均值
    means = {}
    # 遍历第一个样本评估结果中的所有字段（所有样本字段结构相同）
    for key in cleaned_rows[0]:
        if key == "question":
            continue
        # 收集每个指标字段在整个样本评估结果列表中所有非None的值 
        vals = [r[key] for r in cleaned_rows if r[key] is not None]
        # 计算平均值（如果全为None则返回None）
        means[key] = sum(vals) / len(vals) if vals else None
    # 最终结果示例：
    # means = {
    #     "faithfulness": 0.875,
    #     "context_recall": 0.917,
    #     "context_precision": 0.80,
    #     "answer_relevancy": 0.877
    # }

    return {
        "metric_means": means,
        "per_sample": cleaned_rows,
    }



# ═══════════════════════════════════════════════════════════════════════
# 6. 结果输出
# ═══════════════════════════════════════════════════════════════════════

def save_report(
    scores: Dict[str, Any],
    testset: Dataset,
    output_dir: Optional[Path] = None,
) -> Path:
    """将评估结果和测试集保存为带时间戳的 JSON 文件。

    Args:
        scores: evaluate_rag 返回的评估分数
        testset: 测试集（转为 DataFrame 保存，方便人工复查）
        output_dir: 输出目录（None 则默认 OUTPUT_DIR）

    Returns:
        生成的 JSON 文件路径
    """
    # TODO(human): ① output_dir.mkdir(parents=True, exist_ok=True)
    #              ② 构建 report dict（时间戳 / 指标 / 测试样本明细）
    #              ③ json.dump 写入 {output_dir}/eval_report_{timestamp}.json
    # 确定输出目录
    out_dir = output_dir or OUTPUT_DIR
    # 如果父目录不存在则自动创建，目录已存在也不报错
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out_dir / f"eval_report_{timestamp}.json"

    # 构建评估报告
    report = {
        "timestamp": timestamp,
        "metrics": scores.get("metric_means", {}),      # 各指标的均分
        "per_sample": scores.get("per_sample", []),      # 每个样本的评估结果
        "testset": testset.to_pandas().to_dict(orient="records"),  # 测试集的问题和标准答案
    }

    # 将评估结果写入json文件
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)  # 中文字符直接显示不被转义 + JSON 格式化缩进

    print(f"评估报告已保存: {report_path}")
    return report_path


def print_summary(scores: Dict[str, Any]) -> None:
    """在控制台打印评估结果摘要（Markdown 格式）。"""
    # TODO(human): 逐项打印 faithfulness / context_recall / context_precision / answer_relevancy 的均值
    means = scores.get("metric_means", {})
    print("\n" + "=" * 50)
    print("          RAG 评估结果")
    print("=" * 50)
    labels = {
        #    key                            label
        "faithfulness":         "忠实度         (faithfulness)",
        "context_recall":       "上下文召回率   (context_recall)",
        "context_precision":    "上下文精确率   (context_precision)",
        "answer_relevancy":     "答案相关性     (answer_relevancy)",
    }

    for key, label in labels.items():
        # 每个指标的均值数值
        val = means.get(key)
        if val is not None:
            # 保留三位小数。结果示例：忠实度         (faithfulness): 0.875
            print(f" {label}: {val:.3f}")
        else:
            print(f" {label}: N/A")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════════════
# 7. CLI 入口
# ═══════════════════════════════════════════════════════════════════════

def main(test_size: int = DEFAULT_TEST_SIZE):
    """评估主入口：串联 5 步管线。

    Args:
        test_size: 测试样本数（默认 5）
    """
    # TODO(human): ① print 开始横幅
    #              ② docs = _load_parent_docs_from_milvus(n=test_size)
    #              ③ testset = generate_testset(docs, test_size)
    #              ④ rag_service = _init_rag_service()
    #              ⑤ scores = evaluate_rag(rag_service, testset)
    #              ⑥ save_report(scores, testset)
    #              ⑦ print_summary(scores)
    print("╔══════════════════════════════════════════╗")
    print("║     RAG 评估管线 — RAGAS 自动评估       ║")
    print("╚══════════════════════════════════════════╝")
    print(f"  测试样本数: {test_size}")

    try:
        # Step 1: 从 Milvus 随机采样父块
        print("\n── ① 文档采样 ──")
        docs = _load_parent_docs_from_milvus(n=test_size)
        if len(docs) < test_size:
            print(f"警告：实际加载的文档数少于测试样本数{test_size}，仅加载了 {len(docs)} 个文档")
        print(f"   采样完成: {len(docs)} 个文档块")

        # Step 2: RAGAS 自动出题
        print("\n── ② 自动出题 ──")
        testset = generate_testset(docs, test_size=len(docs))
        # testset = generate_testset(docs, test_size)

        # Step 3: 初始化 RagService（BM25 + Vector + RRF）
        print("\n── ③ 初始化 RagService ──")
        rag_service = _init_rag_service()
        print("   RagService 就绪")

        # Step 4: 评估
        print("\n── ④ 评估 ──")
        scores = evaluate_rag(rag_service, testset)

        # Step 5: 保存 + 打印
        print("\n── ⑤ 结果 ──")
        save_report(scores, testset)
        print_summary(scores)

    except Exception as e:
        print(f"\n❌ 评估管线执行失败: {e}")
        import traceback
        traceback.print_exc() # 打印完整错误堆栈信息的函数。


if __name__ == "__main__":
    main()
