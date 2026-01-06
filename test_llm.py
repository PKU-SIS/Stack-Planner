from openai import OpenAI
import os

# os.environ["http_proxy"] = "http://localhost:8888"
# os.environ["https_proxy"] = "http://localhost:8888"
# os.environ["HTTP_PROXY"] = "http://localhost:8888"
# os.environ["HTTPS_PROXY"] = "http://localhost:8888"

# client = OpenAI(
#     base_url="https://api.openai.com/v1",
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# response = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     messages=[{"role": "user", "content": "Test"}],
#     max_tokens=10,
#     temperature=0.0,
# )

client = OpenAI(
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-d47ad54165ee456093bc9ffd599e354e",
)

response = client.chat.completions.create(
    model="qwen3-32b",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=10,
    temperature=0.0,
    extra_body={"enable_thinking": False}
)

# 流式输出响应
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end='', flush=True)

print('\n'*2)  # 输出结束后换行