"""
Step 01: 单轮工具调用 Agent
==============================
目标：用户问"北京天气？"，Agent 调用假的 get_weather 工具，返回结果。
技术点：DeepSeek API 的 function calling + while 循环（ReAct 模式）

运行方式：
    uv run python step01_single_tool_agent.py

前置条件：
    环境变量 DEEPSEEK_API_KEY 已设置
"""

import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI

# ============================================================
# 配置常量
# ============================================================

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
MAX_TURNS = 5  # 最大工具调用轮数，防止死循环

# ============================================================
# 工具定义（OpenAI function calling 格式）
# ============================================================

TOOLS: list[dict[str, Any]] = [
    # TODO(human): 定义 get_weather 工具，包含 name/description/parameters
    # parameters 需要定义 type、properties（location 和 unit）、required
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                    "type": "string",
                    "description": "城市名称",
                },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                    },
                },
            "required": ["location"],
            },
        },
    }
]

# ============================================================
# 工具执行
# ============================================================

async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """根据工具名称分发执行，返回结果字符串。

    Args:
        tool_name: 模型请求调用的工具名称，如 "get_weather"
        arguments: 模型生成的工具参数，如 {"location": "北京", "unit": "celsius"}

    Returns:
        工具执行的结果字符串，会被送回给模型

    Raises:
        ValueError: 工具名称未注册时抛出
    """
    # TODO(human): 实现工具分发逻辑
    # 1. 判断 tool_name 是否为 "get_weather"
    # 2. 从 arguments 中提取 location 和 unit
    # 3. 返回模拟的天气数据（如 "北京当前天气：晴，温度 25°C"）
    if tool_name == "get_weather":
        location = arguments.get("location", "")
        unit = arguments.get("unit", "celsius")

        temp = 25 if unit == "celsius" else 77
        unit_symbol = "摄氏度" if unit == "celsius" else "华氏度"

        return json.dumps({
            "location": location,
            "temperature": temp,
            "unit": unit_symbol,
            "condition": "qingtian",
            "humidity": "45%",
        }, ensure_ascii=False)

    raise ValueError(f"unknown tools")

# ============================================================
# Agent 核心循环（ReAct 模式）
# ============================================================

async def run_agent(user_input: str, client: AsyncOpenAI) -> str:
    """Agent 主循环：思考 → 调用工具 → 观察结果 → 返回答案。

    循环逻辑：
    1. 将用户输入放入 messages 列表
    2. 调用模型，传 messages + tools
    3. 如果模型直接回复文本（无 tool_calls），返回给用户
    4. 如果模型要调用工具，执行工具，把结果附加到 messages，回到步骤 2
    5. 达到 MAX_TURNS 上限时强制退出

    Args:
        user_input: 用户输入的问题
        client: 已配置好的 AsyncOpenAI 客户端

    Returns:
        Agent 的最终文本回复
    """
    # TODO(human): 实现 Agent 循环
    # 1. 初始化 messages 列表（第一条是用户消息）
    # 2. while 循环（用 turn_count 计数，上限 MAX_TURNS）
    # 3. 调用 client.chat.completions.create(model=..., messages=..., tools=..., tool_choice="auto")
    # 4. 取 response.choices[0].message
    # 5. 如果 message.tool_calls 为空 → return message.content
    # 6. 如果有 tool_calls → 逐个执行，附加 tool role 消息到 messages，继续循环

# ============================================================
# 命令行交互入口
# ============================================================

async def main() -> None:
    """命令行交互入口：初始化客户端 → 接收用户输入 → 调用 Agent → 打印结果。"""
    # TODO(human): 实现主函数
    # 1. 读取 DEEPSEEK_API_KEY 环境变量，未设置时给出友好提示并退出
    # 2. 创建 AsyncOpenAI 客户端（base_url + api_key）
    # 3. 提示用户输入问题
    # 4. 调用 run_agent() 并打印结果
    # 5. 处理可能出现的异常（API 超时、网络错误等）
