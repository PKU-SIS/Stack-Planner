from openai import OpenAI

api_key = "sk-550b5def797e452184c320b368a10989"
client = OpenAI(api_key=api_key, base_url="http://123.57.228.132:8286/api")

message_content='''Why Sky is Blue''


response = client.chat.completions.create(
    model="deepseek-v3.2-20251201-160k-local",
    messages=[{"role": "user", "content": message_content}],
)

response = response.choices[0].message.content
print(response)


