
from openai import OpenAI
client = OpenAI(
  api_key="1", 
  base_url="http://10.1.1.212:8000/v1")

response = client.chat.completions.create(
  model="Qwen3-32B",
  messages=[{"role": "user", "content": "Why is the sky blue?"}],
  stream=True,
)

# 流式输出响应
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end='', flush=True)

print('\n'*2)  # 输出结束后换行