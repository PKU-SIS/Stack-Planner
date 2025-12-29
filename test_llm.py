
from openai import OpenAI
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
