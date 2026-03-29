# utils.py
from openai import OpenAI
# API_CONFIG = {
#     "base_url": "http://123.57.228.132:8286/api",
#     "api_key": "sk-550b5def797e452184c320b368a10989",
#     "model": "deepseek-v3.2-20251201-160k-local",
# }
# API_CONFIG = {
#     "base_url": "http://10.1.1.212:8000/v1",
#     "api_key": "sk-d47ad54165ee456093bc9ffd599e354e",
#     "model": "Qwen3-32B",
# }
API_CONFIG = {
    "base_url": "http://123.59.6.244:8000/v1",
    "api_key": "sk-d47ad54165ee456093bc9ffd599e354e",
    "model": "Qwen3-32B",
}



# 初始化 client
client = OpenAI(
    base_url=API_CONFIG["base_url"],
    api_key=API_CONFIG["api_key"]
)

def llm(prompt, model=None):
    if model is None:
        model = API_CONFIG["model"]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def main():
    prompt = "请简单介绍一下大语言模型的基本原理。"
    
    print("Prompt:")
    print(prompt)
    print("\nResponse:")

    result = llm(prompt)
    print(result)


if __name__ == "__main__":
    main()