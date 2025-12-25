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
    base_url="http://123.57.228.132:8285",
    api_key="sk-d47ad54165ee456093bc9ffd599e354e",
)

response = client.chat.completions.create(
    model="Qwen2.5-32B-Instruct",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=10,
    temperature=0.0,
)

print(f"📝 Response: {response.choices[0].message.content}")
