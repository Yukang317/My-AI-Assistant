#!/usr/bin/env python3
"""批量灌库：把 dev_log/ 和 docs/ 下的 Markdown 走 DocumentIndexer 入库。"""

from __future__ import annotations

import argparse             # 命令行参数解析
import logging
import sys
import tempfile             # 创建临时文件 / 目录（测试用）
from pathlib import Path    # 路径操作
from typing import Any      # 类型提示：Any 表示任意类型
from unittest.mock import MagicMock, patch     # 单元测试 Mock 工具
# - MagicMock：创建一个万能假对象，调用它的任何方法都不会报错，返回值可以自定义
# - patch：临时替换掉真实的函数/类，测试完自动恢复



ROOT = Path(__file__).resolve().parent.parent   # ROOT = personal_assistant/
if str(ROOT) not in sys.path:           # 如果 ROOT 不在搜索路径里
    sys.path.insert(0, str(ROOT))       # 把 ROOT 加到搜索路径的最前面

import db
from rag.indexer import DocumentIndexer

logger = logging.getLogger(__name__)

DISCUSSION_DIR = "docs/讨论过程"


def collect_markdown_files(root: Path, include_discussion: bool = False) -> list[Path]:
    """收集待入库的 Markdown 文件路径（已排序、去重）。

    Args:
        root: 项目根目录 personal_assistant/
        include_discussion: 是否包含 docs/讨论过程/

    Returns:
        相对 root 的 .md 文件路径列表
    """
    # TODO(human): 实现文件扫描逻辑
    found: set[Path] = set()    # 用集合去重

    dev_log_dir = root / "dev_log"  # 拼接：personal_assistant/dev_log/
    if dev_log_dir.is_dir():        # 目录不存在就跳过
        found.update(dev_log_dir.rglob("*.md"))   # 递归获取所有 .md 文件，加入 set

    docs_dir = root / "docs"
    if docs_dir.is_dir():
        found.update(docs_dir.glob("*.md"))
        if include_discussion:      # 只有用户明确传 --include-discussion 才收讨论过程（开关控制）
            discussion_dir = docs_dir / "讨论过程"
            if discussion_dir.is_dir():
                found.update(discussion_dir.rglob("*.md"))
    
    return sorted(p.resolve() for p in found)   # 相对路径转绝对路径，按字符串排序




def index_one_file(
    indexer: DocumentIndexer,
    file_path: Path,
    root: Path,
) -> dict[str, str | int]:
    """读取单个文件并调用 DocumentIndexer 入库，必要时写入 PG。

    Args:
        indexer: DocumentIndexer 实例
        file_path: 待入库文件的绝对路径
        root: 项目根目录

    Returns:
        {"status": "new"|"updated"|"skipped"|"failed", "relative": str, ...}
    """
    # TODO(human): 实现单文件入库逻辑
    # 1. 准备数据
    relative = file_path.relative_to(root).as_posix()   # Path("dev_log/阶段5/xxx.md") -> "dev_log/阶段5/xxx.md" (强制用 /)
    file_data = file_path.read_bytes()                  # 读取文件字节流，用于 Indexer 的 MD5 计算和 MinIO 上传

    # 2. 调用 indexer
    result = indexer.index_document(file_data, relative)
    # result = {
    #     "status":    "new" | "skipped" | "updated",   ← 三种结果
    #     "object_key": "documents/2026/06/dev_log/xxx.md",
    #     "file_md5":  "abc123...",
    #     "file_type": ".md",
    #     "parent_count": 3,
    #     "child_count": 12,
    #     "message": "...",
    # }
    status = result["status"]

    # 3. 分支处理
    if status == "skipped":
        return {
            "status": "skipped",
            "relative": relative,
            "object_key": result.get("object_key", ""),
            "message": result.get("message", ""),
        }
    
    # 4. 写 PostgreSQL 台账
    db.save_document(
        filename=relative,                      # 相对路径，如 "dev_log/阶段5/xxx.md"
        file_type=result["file_type"],          # ".md"
        file_size=len(file_data),               # 原始字节数，如 12345
        object_key=result["object_key"],        # Min IO 对象 key，如 "documents/2026/06/dev_log/xxx.md"
        file_md5=result["file_md5"],            # MD5 值，如 "abc123..."
        parent_count=result["parent_count"],    # 父块数量
        child_count=result["child_count"],      # 子块数量
    )

    # 5. 返回给上层
    return {
        "status": status,
        "relative": relative,
        "object_key": result["object_key"],
        "parent_count": result["parent_count"],
        "child_count": result["child_count"],
    }





def run_seed(
    root: Path,
    *,          # 分隔符，左边按位置传，右边必须写参数名
    dry_run: bool = False,
    include_discussion: bool = False,
) -> dict[str, int]:
    """执行批量灌库，返回统计摘要。

    Args:
        root: 项目根目录
        dry_run: True 时只列出文件，不实际入库
        include_discussion: 是否包含 docs/讨论过程/

    Returns:
        {"total": N, "new": X, "updated": Y, "skipped": Z, "failed": F}
    """
    # TODO(human): 实现批量调度与统计
    # 文件路径与统计结果准备
    files = collect_markdown_files(root, include_discussion=include_discussion)
    stats = {
        "total": len(files),
        "new": 0,
        "updated": 0,
        "skipped": 0,
        "failed": 0,
    }
    
    # 干跑：“给我看看你会处理哪些文件，但别真的动数据库”
    if dry_run:
        for file_path in files:
            print(file_path.relative_to(root).as_posix())
        return stats    # total 有值，其他字段全是 0
    
    # 初始化数据库和索引器
    db.init_db()    # 建表（如没有）
    indexer = DocumentIndexer()     # 加载 BGE 模型到内存
    
    # 逐文件处理循环
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        try:
            outcome = index_one_file(indexer, file_path, root)
            status = outcome["status"]
            if status in stats:    # 如果 Indexer 某天新增了一种 status（比如 "error"），它不会因为 KeyError 炸掉，而是静默忽略。
                stats[status] += 1
            print(
                f"[{status.upper():7}] {relative} "
                f"({outcome.get('parent_count', '-')}p / {outcome.get('child_count', '-')}c)"
            )
        except Exception as exc:
            stats["failed"] += 1
            logger.exception("索引失败: %s", relative)
            print(f"[FAILED ] {relative}: {exc}")
    
    return stats

def main() -> None:
    """CLI 入口。"""
    # TODO(human): 实现 argparse + 调用 run_seed
    parser = argparse.ArgumentParser(description="批量灌库 dev_log/ + docs/ Markdown 到 RAG")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出待入库文件，不连接 Milvus/MinIO",
    )
    parser.add_argument(
        "--include-discussion",
        action="store_true",
        help=f"额外包含 {DISCUSSION_DIR}/ 下的 md（默认排除）",
    ) # 命令行加对应参数就是 True，不加就是 False

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    stats = run_seed(
        ROOT,
        dry_run=args.dry_run,
        include_discussion=args.include_discussion,
    )

    if args.dry_run:
        print(f"\n干跑完成：共 {stats['total']} 个文件待入库")
    else:
        print(
            f"\n灌库完成：共 {stats['total']} 个文件，"
            f"新增 {stats['new']}，更新 {stats['updated']}，"
            f"跳过 {stats['skipped']}，失败 {stats['failed']}"
        )


if __name__ == "__main__":
    main()
