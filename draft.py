


# from openai import OpenAI

# api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"
# client = OpenAI(api_key=api_key, base_url="http://123.57.228.132:8285/api")

# completion = client.chat.completions.create(
#     model="deepseek-v3.1-160k-local",
#     messages=[{"role": "user", "content": "Why is the sky pink?"}],
#     stream=True
# )

# for chunk in completion:
#     if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")
# print()

# import requests

# url = "https://ragflow.pkubir.cn/v1/kb_api/list"
# params = {
#     "page": 1,
#     "page_size": 20,
#     "keywords": "",
#     "orderby": "create_time",
#     "desc": "true"
# }
# data = {
#     "tenant_id": "cbae14fb8c8411f0bf2ecd6543f8a381"  #这里提供的子然账号，XXQG知识库在这上面
# }

# response = requests.post(url, params=params, json=data)
# result = response.json()

# if result["code"] == 0:
#     kbs = result["data"]["kbs"]
#     total = result["data"]["total"]
#     print(f"获取到 {total} 个知识库")
#     for kb in kbs:
#         print(f"- {kb['name']} (ID: {kb['id']})")
# else:
#     print(f"Error: {result.get('message', 'Unknown error')}")

import requests

url = "https://ragflow.pkubir.cn/v1/chunk_api/retrieval_test"
headers = {
    "Content-Type": "application/json"
}

# 基础检索
data = {
    "tenant_id": "cbae14fb8c8411f0bf2ecd6543f8a381",      # zzr账号
    "kb_id": ["75d78910a00911f0bf2ecd6543f8a381"],            # XXQG知识库，745篇文档，习总书记相关
    "question": "文化八项工程",
    "page": 1,
    "size": 10
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
print("result",result)
# if result["code"] == 0:
#     total = result["data"]["total"]
#     chunks = result["data"]["chunks"]
#     labels = result["data"].get("labels", [])

#     print(f"检索到 {total} 个相关chunks")
#     print(f"标签: {labels}\n")

#     for i, chunk in enumerate(chunks, 1):
#         print(f"Chunk {i}:")
#         print(f"  相似度: {chunk.get('similarity', 0):.4f}")
#         print(f"  文档: {chunk['docnm_kwd']}")
#         print(f"  内容: {chunk['content_with_weight'][:150]}...")
#         print()
# else:
#     print(f"Error: {result.get('message', 'Unknown error')}")
