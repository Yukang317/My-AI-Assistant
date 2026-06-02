"""
================================================================================
第 1 步：调通 DeepSeek API — 纯 Python 脚本
目标：用一个最简单的脚本，验证能成功调用 DeepSeek 大模型并拿到回复。
"""

# 第一部分：导入依赖
import os
from dotenv import load_dotenv
from openai import OpenAI

# 第二部分：加载配置
load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = "deepseek-chat"

# 第三部分：创建客户端
# OpenAI(...) 创建一个"客户端对象"，后续所有和大模型的交互都通过这个对象进行
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# 第四部分：发送消息
# 错误处理
# try:
#     # 构造一个 HTTP 请求，发送到 DeepSeek 服务器，然后等待回复
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "你是一个乐于助人的个人助理，用中文回答所有问题。"},
#             {"role": "user", "content": "你好！请用一句话介绍一下你自己。"},
#         ],
#         temperature=0.7,
#     )
# # 第五部分：打印回复
#     reply_text = response.choices[0].message.content
#     print(f"🤖 DeepSeek 回复：{reply_text}")
# except Exception as e:
#     print(f"发生错误：{e}")

messages =[
    {"role":"system","content":"你是一个乐于助人的个人助理，用中文回答所有问题"},
    ]

while True:

    question = input("👤你：")

    if question == "退出":
        print("👋 再见！")
        print(messages)
        break
    
    messages.append({"role":"user","content":question})
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        reply_content = response.choices[0].message.content


        messages.append({"role": "assistant", "content": reply_content})
        print(f"🤖 助手：{reply_content}")
    except Exception as e:
        print(f"发生错误：{e}")