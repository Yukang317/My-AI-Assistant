"""
Exa Search 简单连通性测试

运行：
  cd personal_assistant && PYTHONPATH=. uv run python test/exa_search_test.py

需要 .env 中配置 EXA_API_KEY；未配置时自动跳过。
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from agent.tools.base import ToolContext
from agent.tools.web_search import exa_search


def main() -> int:
    if not os.getenv("EXA_API_KEY"):
        print("SKIP: 未配置 EXA_API_KEY，跳过 Exa 测试")
        return 0

    ctx = ToolContext(session_id="test-exa-session")
    query = "What is LangGraph AI agent framework"

    print("=" * 50)
    print(f"Exa 搜索: {query}")
    print("=" * 50)

    result = exa_search(ctx, query, max_results=3)

    print(f"success: {result.success}")
    if not result.success:
        print(f"error: {result.error}")
        return 1

    results = (result.artifacts or {}).get("results", [])
    print(f"结果数: {len(results)}")
    print(f"内容长度: {len(str(result.data))} 字符")
    print("\n前 2 条来源:")
    for i, item in enumerate(results[:2], 1):
        print(f"  {i}. {item.get('title', '')}")
        print(f"     {item.get('url', '')}")

    print("\n内容预览:")
    print(str(result.data)[:300], "...")
    print("\nExa 测试通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
