from openai import OpenAI

api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"
client = OpenAI(api_key=api_key, base_url="http://162.105.88.35:3000/api")

completion = client.chat.completions.create(
    model="deepseek-v3.1-160k-local",
    messages=[{"role": "user", "content": "Why is the sky pink?"}],
    stream=True
)

for chunk in completion:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()