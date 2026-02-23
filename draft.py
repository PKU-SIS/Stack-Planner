from openai import OpenAI

api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"
client = OpenAI(api_key=api_key, base_url="http://162.105.88.35:3000/api")

response = client.chat.completions.create(
    model="deepseek-v3.2-20251201-160k-local",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)

response = response.choices[0].message.content
print(response)
