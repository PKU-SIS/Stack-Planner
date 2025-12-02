# from openai import OpenAI

# api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"
# client = OpenAI(api_key=api_key, base_url="http://162.105.88.35:3000/api")

# completion = client.chat.completions.create(
#     model="deepseek-v3.1-160k-local",
#     messages=[{"role": "user", "content": "Why is the sky pink?"}],
#     stream=True
# )

# for chunk in completion:
#     if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")
# print()


# from openai import OpenAI

api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"

import time
from openai import OpenAI

start_time = time.time()

client = OpenAI(api_key=api_key, base_url="http://60.28.106.46:8289/v1")


question='介绍一下牛顿定律'
full_text = "qwen3-32b"
with client.chat.completions.stream(
    model="llama-3.2-1b-instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user","content":question }#  "帮我写一段测试代码。"
    ],
) as stream:
    for event in stream:
        if event.type == "content.delta":
            # 每次拿到增量内容
            delta = event.delta
            print(delta, end="", flush=True)  # 实时打印
            full_text += delta               # 拼接到结果里

        elif event.type == "content.done":
            # 流式输出结束标志
            print("\n--- 流式输出结束 ---")


# 结束计时
end_time = time.time()
elapsed = end_time - start_time

# 统计字符长度
char_len = len(full_text)
# 计算平均 token/s
if elapsed > 0:
    tokens_per_sec = char_len / elapsed
else:
    tokens_per_sec = 0

# 输出结果
print(f"\n⏱️ 总耗时：{elapsed:.2f} 秒")
print(f"📝 输出总长度：{char_len} 字符")
print(f"⚡ 平均生成速度：{tokens_per_sec:.2f} token/s")
