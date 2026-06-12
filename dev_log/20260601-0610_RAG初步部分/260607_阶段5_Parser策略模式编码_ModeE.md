# 阶段 5 — rag/parser/ 策略模式编码（Mode E 骨架 → 逐个实现）

**日期**：2026-06-07（第 9 次会话，编码部分）
**模式**：Mode E（骨架式教学开发）
**内容**：生成 6 个 parser 文件骨架，逐函数教学，用户实现所有 TODO(human)

---

## 〇、做了什么 & 为什么

**当前位置**：阶段 5 RAG，路线 1（文档摄入）。Step 1-4（config / embedding / minio_client / vector_store）已完成，这次做 Step 5 — 文档解析器。

**本次完成了什么**：
1. 生成 6 个文件的完整骨架（import + class/func signatures + docstring + TODO(human)）
2. 讲解策略模式、"parser 为什么薄"、parser vs chunker 的职责边界
3. 用户逐个实现所有 TODO(human)，6 个文件全部编译通过
4. 深入讨论了策略模式的价值（消除 if/elif、未来加新格式不改 coordinator）

**这个模块在架构中的作用**：Parser 是文档摄入管线的第一站。它把 PDF/DOCX/MD/TXT 四种不同格式统一转换成纯文本字符串，消除格式差异，让下游 chunker 只需面对一种输入。Parser 自身不做"理解文本"的事——那是 chunker 的职责。

---

## 一、文件清单 & 设计

### 6 个文件的分工

```
rag/parser/
├── __init__.py          # 包说明
├── base.py              # DocumentParser 抽象基类（定义 parse/supports 接口）
├── pdf_parser.py        # PDF → Markdown（pymupdf4llm）
├── docx_parser.py       # Office → Markdown（markitdown）
├── markdown_parser.py   # .md/.markdown 多编码解码
├── text_parser.py       # 纯文本兜底解析器
└── coordinator.py       # 策略模式协调器（平面注册 + _get_parser + parse）
```

### 设计原则

| 决策 | 选择 | 原因 |
|------|------|------|
| 架构 | 策略模式 | 每个解析器一个类，Coordinator 遍历匹配 |
| PDF 方案 | pymupdf4llm | 底层 PyMuPDF（C 库），保留表格/标题/图片 |
| Office 方案 | markitdown | 一个库覆盖 doc/docx/xls/xlsx/ppt/pptx |
| MD/TXT | 多编码 fallback | utf-8 → gbk → gb2312 → latin-1 兜底 |
| 按需导入 | import 在 parse() 内 | __init__ 不加载重型库（pymupdf/transformers），省内存 |
| 临时文件 | NamedTemporaryFile | PDF/DOCX 解析库需要文件路径而非 bytes |
| 兜底策略 | TextParser 放最后 | supports() 宽泛，任何未知格式至少能尝试解码 |

---

## 二、核心实现要点

### base.py — 抽象基类

```python
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, data: bytes) -> str: ...
    @abstractmethod
    def supports(self, file_type: str) -> bool: ...
```

只有 2 个抽象方法，接口极简。Coordinator 不需要知道具体是哪个解析器。

### pdf_parser.py — pymupdf4llm

关键点：
- `NamedTemporaryFile(suffix=".pdf", delete=False)` — 后缀必须是 .pdf，否则 pymupdf4llm 不认
- 用 `Path(tmp.name).write_bytes(data)` 写入字节
- `finally` 块中 `os.unlink(path)` 清理临时文件
- `FileNotFoundError` 静默忽略（库内部可能已清理）

### docx_parser.py — markitdown

关键点：
- `MarkItDown().convert(source_path)` 返回 `ConversationResult`
- 取 `.text_content` 获取 Markdown 文本
- 后缀用 `.tmp` 即可，markitdown 自动检测格式

### markdown_parser.py / text_parser.py — 多编码 fallback

```python
ENCODINGS = ["utf-8", "gbk", "gb2312", "latin-1"]

def parse(self, data: bytes) -> str:
    for encoding in self.ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")  # 永不抛异常
```

`latin-1` 是单字节全覆盖编码（0x00-0xFF 全部合法），所以不会抛 UnicodeDecodeError。

### coordinator.py — 策略模式协调器

关键点：
- `_get_parser()` 遍历 `self.parsers`，返回第一个 `supports()` 返回 True 的
- TextParser 放列表最后 → 自然兜底
- `supported_types()` 收集所有解析器的 `SUPPORTED_TYPES` 并去重排序

---

## 三、关键讨论：Parser 为什么"薄"

用户提出深刻困惑：parser 只做简单调用（pymupdf4llm/markitdown 或解码），感觉没写什么有价值的东西。

**结论**：Parser 确实薄——因为它的职责被严格限定为"消除格式差异"：

```
Parser 的事：           Chunker 的事（Step 6）：
  PDF bytes → 文本         Markdown 文本 → 识别 H1-H6 标题层级
  DOCX bytes → 文本                       → 按段落边界切分
  MD 文件 → 解码                          → 构建父子块
  TXT 文件 → 解码
```

**Parser 薄不是缺陷**。PDF/DOCX 的二进制解析复杂度被 pymupdf4llm 和 markitdown 吃掉了（底层分别是 C 写的 PyMuPDF 和 python-docx/openpyxl），MD/TXT 本来就不需要格式转换。真正"理解文本结构"的复杂度被故意推到了 chunker 层。

策略模式在此处的价值不是让 parse 变复杂，而是：
1. 消除 if/elif 分支
2. 未来加 HTML/图片 OCR 解析器时，写一个新文件 + 在 coordinator 列表加一行即可
3. Coordinator 不需要知道哪个解析器在处理——调用方只需 `coordinator.parse(data, file_type)`

---

## 四、编译验证

```bash
$ python -m py_compile rag/parser/*.py
全部通过（6/6）
```

---

## 五、知识点汇总

| # | 知识点 | 对应文件 |
|---|--------|---------|
| 1 | 策略模式（Strategy Pattern）：接口定义 + 多实现 + 协调器遍历匹配 | base.py + coordinator.py |
| 2 | 按需导入（lazy import）：重型库不在 __init__ 加载，在 parse() 内 import | pdf_parser.py, docx_parser.py |
| 3 | pymupdf4llm：PyMuPDF C 库的 Python 包装，PDF → Markdown 一键转换 | pdf_parser.py |
| 4 | markitdown：微软开源，一个库覆盖所有 Office 格式 → Markdown | docx_parser.py |
| 5 | 多编码 fallback：utf-8 → gbk → gb2312 → latin-1（latin-1 永不抛异常） | markdown_parser.py, text_parser.py |
| 6 | 临时文件生命周期：NamedTemporaryFile(delete=False) + finally os.unlink() | pdf_parser.py, docx_parser.py |
| 7 | Path.write_bytes() / Path.read_bytes() — Python 3.8+ 的简洁文件操作 | pdf_parser.py, docx_parser.py |
| 8 | Parser vs Chunker 职责边界：格式消除 vs 结构理解 | 架构讨论 |
| 9 | 策略模式 vs if/elif：用类的形式组织分支，开闭原则（对扩展开放，对修改关闭） | coordinator.py |
