# 阶段 5 — Embedding 模块（rag/embedding.py）

**日期**：2026-06-02
**模式**：C（用户手写代码，Claude review + 讲解概念）
**内容**：实现 `rag/embedding.py` 的完整代码，深入理解 Python 内存管理、BGE 向量化参数、GPU vs CPU 部署行业标准

---

## 〇、做了什么 & 为什么

### 当前进度

阶段 5 的 Step 1（`rag/config.py`）已完成，Step 2（`rag/embedding.py`）在上次会话已搭好构造函数骨架。本次会话完成整个文件的编写和 review。

### embedding.py 在 RAG 系统的位置

```
文档摄入（路线1）：文档 → 解析 → 分块 → 【EmbeddingService.embed()】→ 写 Milvus
知识检索（路线2）：用户问题 → 【EmbeddingService.embed()】→ 搜 Milvus → 重排 → LLM
```

它是**两条路线的共用基础设施**——不管是建索引还是搜知识，都要先把文本变成向量。

---

## 一、代码全貌

```python
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
        self.model = None    # local 模式的模型实例
        self.client = None   # api 模式的 OpenAI client

        if mode == "local":
            self._init_local_model()
        elif mode == "api":
            self._init_api_client()
        else:
            raise ValueError(f"不支持的 embedding 模式：{mode}")

    # ── 本地模式 ──────────────────────────────
    def _init_local_model(self):
        """加载 BGE 模型到内存"""
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            model_name=Config.EMBEDDING_LOCAL_MODEL,
            device="cpu"
        )

    # ── API 模式 ──────────────────────────────
    def _init_api_client(self):
        """初始化 OpenAI 兼容的 embedding 客户端"""
        from openai import OpenAI
        self.client = OpenAI(
            api_key=Config.EMBEDDING_API_KEY,
            base_url=Config.EMBEDDING_API_BASE,
        )

    # ── 核心方法 ──────────────────────────────
    def embed(self, texts: List[str]) -> List[List[float]]:
        """把文本列表转成向量列表"""
        if self.mode == "local":
            # 记得开归一化
            return self.model.encode(texts, normalize_embeddings=True).tolist()
        elif self.mode == "api":
            # TODO: API 模式暂未实现
            raise NotImplementedError("API 模式暂未实现")
        else:
            raise ValueError(f"不支持的 embedding 模式：{self.mode}")

    # ── 资源管理 ──────────────────────────────
    def unload(self):
        """释放模型内存（索引完成后调用，日常聊天不驻留）"""
        if self.model is not None:
            del self.model      # 引用计数归零 → 立刻释放大部分内存（~1.3GB）
            self.model = None   # 确保引用被清除
            import gc
            gc.collect()
```

---

## 二、自提疑问 & 解答（本次核心讨论）

### Q1：`gc.collect()` 是干什么的？为什么 `del` 之后还要手动调用它？

**背景**：Python 有两层内存回收机制：

| 机制 | 工作原理 | 触发时机 | 局限 |
|------|---------|---------|------|
| **引用计数** | 每个对象记录有多少变量指向它。归零 → 立即释放内存 | 实时，`del` 的瞬间 | **处理不了循环引用**（A 引用 B，B 引用 A，两个计数都不归零） |
| **GC（垃圾回收）** | 定期扫描所有对象，找到"存在但不可达"的循环引用孤岛 → 回收 | 自动触发，时机不确定 | 可能延迟很久才触发 |

```python
del self.model      # 引用计数归零 → 立刻释放大部分内存（~1.3GB）
self.model = None   # 确保引用被清除（虽然 del 就够了，防御性写法）
```

但 BGE 模型内部有大量嵌套对象（tokenizer、model、config 等），互相之间存在引用。虽然 SentenceTransformer 内部没有刻意制造循环引用，但大模型库的内部对象关系复杂，不排除有隐蔽的循环引用。

`gc.collect()` 的作用：**手动触发一次完整的垃圾回收扫描**，把循环引用也清理掉。

> **类比**：你扔掉了垃圾桶（`del`），但垃圾桶里的塑料袋可能还没被垃圾车收走（GC）。`gc.collect()` 就是手动通知垃圾车立刻来一趟。

**实际必要性**：对于这个场景，`del self.model` 通常就够了。`gc.collect()` 是一种**防御性措施**——多花 0.01 秒，确保 1.3GB 内存彻底释放，不给 ECS 3.5GB 小内存留隐患。

---

### Q2：`model.encode(texts, normalize_embeddings=True)` 两个参数是什么意思？

#### 参数一：`texts` — 输入文本

```python
# 单条
model.encode("你好世界")           # → numpy array shape: (1024,)

# 批量（我们代码里用这种）
result = model.encode(["文本1", "文本2", "文本3"])
# result 是 numpy ndarray，shape 为 (3, 1024)，即 3 个文本，每个 1024 维

# .tolist() 把 numpy 转成 Python 原生 list
# [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
```

**为什么要传列表而不是一条条调**：批量处理比逐条快很多——模型对 batch 内部做矩阵运算，充分利用 CPU 的向量指令集（SIMD）。

#### 参数二：`normalize_embeddings=True` — L2 归一化

**这条非常重要，BGE 官方文档要求必须开。**

```
原始向量：   [0.3, 0.8, 0.1, ..., 0.5]    ← L2 范数（模长）= 0.86
归一化后：   [0.35, 0.93, 0.12, ..., 0.58] ← L2 范数 = 1.0（每个值除以模长）
```

**为什么要归一化？跟 Milvus 的索引类型直接相关：**

| 索引类型 | 计算方式 | 适用条件 |
|---------|---------|---------|
| **IP（内积）** | 直接算 A·B | **要求向量已归一化**（此时内积 = 余弦相似度） |
| COSINE | 算 A·B / (|A|×|B|) | 不要求归一化，但计算量大 |

用归一化向量 + Milvus IP 索引：搜索速度最快，精度一样。

> **数学原理**：余弦相似度 = A·B / (|A|×|B|)。如果 |A| = |B| = 1（归一化后），则余弦相似度 = A·B（就是内积）。省掉了除法，每次搜索都更快。

---

### Q3：ECS 没有 GPU，我的笔记本有 GPU。能不能在笔记本上跑 embedding，把结果传到 ECS 上？

**这是我这次最想问的问题——行业里到底怎么做？**

#### Claude 的回答：不要这样做，用 CPU 就够了

**"本地 GPU 算完传服务器"不是行业标准做法**，原因有三：

1. **管道断裂**：上传文档 → 传到笔记本 GPU → 向量化 → 传回 ECS → 写 Milvus。本应一条命令的事变成了跨机器的复杂流程。

2. **增量索引难搞**：以后每次上传一篇新文档，都要走"ECS → 你的笔记本 → ECS"的往返。笔记本不开机，整个 RAG 系统就瘫痪。

3. **笔记本不是服务器**：开发阶段你可能在笔记本上操作，但以后这套系统应该 7×24 独立运行在 ECS 上。

#### 行业标准做法：三档方案

```
┌────────────────┬──────────────────────────────┬──────────────┐
│  规模          │  方案                         │  月成本       │
├────────────────┼──────────────────────────────┼──────────────┤
│  个人/开发     │  CPU 本地推理（跟服务同机部署） │  0 元         │
│  小团队/创业   │  Embedding API 服务            │  几十~几百    │
│  企业/大规模   │  专用 GPU 推理服务（如 TEI）    │  几千+        │
└────────────────┴──────────────────────────────┴──────────────┘
```

#### BGE-large-zh-v1.5 在 CPU 上的实际性能

| 场景 | 耗时 | 感受 |
|------|------|------|
| 批量索引 100 篇文档 | 2~5 分钟 | 一次性，等得起 |
| 增量索引 1 篇新文档 | 2~5 秒 | 完全可接受 |
| 查询时向量化（1 个问题） | 0.1~0.3 秒 | 用户无感知 |

**结论**：个人知识库这个场景，CPU 绰绰有余。索引是一次性的，查询频率低（一天几十次顶天了）。你看到的大量教程让你用 GPU，那是针对企业生产环境（每秒几百次查询）。

#### 以后真需要 GPU 时的正确方案

不是"用笔记本 GPU"，而是：

1. **短期**：把 `EMBEDDING_MODE` 切到 `api`，用阿里云/OpenAI 的 embedding API，按量付费，几块钱一个月
2. **长期**：在阿里云上开 GPU 实例，或直接用阿里云的模型服务托管 BGE

> 这两种都是行业标准做法，不需要笔记本参与。**服务应该自己能跑，不依赖开发者的个人电脑。**

---

## 三、代码 Review 过程 & 踩坑记录

### 本轮发现的 Bug（共 6 个）

| # | Bug | 原因 | 修复 |
|---|-----|------|------|
| 1 | `model_name=..., device=...` 缺逗号 | Python 语法错误 | 加逗号 |
| 2 | `EMBEDDING_API_URL` 不存在 | 拼写错误，config.py 里叫 `EMBEDDING_API_BASE` | 改成 `EMBEDDING_API_BASE` |
| 3 | `def embed(self, text: List[str])` | 参数名叫 `text` 但传的是列表 | 改成 `texts: List[str]` |
| 4 | `raise ValueError("不支持的...{mode}")` | 漏了 f-string 前缀 | 加 `f` |
| 5 | `if self.mode is not None` | 应该判断 `self.model` 而不是 `self.mode` | 改成 `self.model is not None` |
| 6 | `elif self.mode == "api":` 后面的 `raise ValueError` 缩进错了 | `raise` 写在 elif 块里面（12 空格缩进），导致合法的 "api" 模式也被拒 | 取消注释 `else:`，各分支分离 |

**Bug 6 的详细分析**：

```python
# ❌ 错误代码
        elif self.mode == "api":
        #     response = self.client...
        # else:
            raise ValueError(f"不支持的 embedding 模式：{self.mode}")
#                         ↑ 这行有 12 空格缩进 → 属于 elif 块！
#                           当 mode="api" 时走到这里，把合法模式拒绝

# ✅ 正确代码
        elif self.mode == "api":
            raise NotImplementedError("API 模式暂未实现")
        else:
            raise ValueError(f"不支持的 embedding 模式：{self.mode}")
```

这是典型的**缩进导致的作用域错误**——注释掉的代码行混在中间，眼睛容易看错缩进层级。

---

## 四、协作模式评估

本次使用 **C 模式**（Claude 给骨架 + 讲解，用户写代码）。评价：

- ✅ 类结构、双模式分支、`embed()` 和 `unload()` 自己写出了
- ✅ 能主动加注释解释 `del` + `gc.collect()` 的用途
- ⚠️ 有 6 个小 bug，但都是能被 review 发现的（非逻辑错误）
- 追问的三个问题（gc、encode 参数、GPU vs CPU）都切中要害，说明在深入理解代码而非照抄

**下一步建议**：embedding.py 写完后，进入 Step 3 `minio_client.py`。MinIO SDK 不熟悉，切 **A 模式**（Claude 写完整代码，用户读 + 加注释）。

---

## 五、文件状态

| 文件 | 状态 | 说明 |
|------|------|------|
| `rag/config.py` | ✅ | 8 组配置，验证通过 |
| `rag/embedding.py` | ✅ | Step 2 完成（还有缩进 bug 待修，见上表 #6） |
| `rag/storage/minio_client.py` | ⬜ | Step 3，下一步 |
