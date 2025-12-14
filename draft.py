

# api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjkwZDBjNmU0LTI1MzUtNGQ3OS1hOGI4LWUyMGJmYzIwMmIwYSJ9.xCJO76Cj2OMoEo1du9NTj0BI_wZIfYezCk3zbiijjqM"

# import time
# from openai import OpenAI

# start_time = time.time()

# client = OpenAI(api_key=api_key, base_url="http://60.28.106.46:8289/v1")


# question='介绍一下牛顿定律'
# full_text = "qwen3-32b"
# with client.chat.completions.stream(
#     model="llama-3.2-1b-instruct",
#     messages=[
#         {"role": "system", "content": "你是一个有帮助的助手。"},
#         {"role": "user","content":question }#  "帮我写一段测试代码。"
#     ],
# ) as stream:
#     for event in stream:
#         if event.type == "content.delta":
#             # 每次拿到增量内容
#             delta = event.delta
#             print(delta, end="", flush=True)  # 实时打印
#             full_text += delta               # 拼接到结果里

#         elif event.type == "content.done":
#             # 流式输出结束标志
#             print("\n--- 流式输出结束 ---")


# # 结束计时
# end_time = time.time()
# elapsed = end_time - start_time

# # 统计字符长度
# char_len = len(full_text)
# # 计算平均 token/s
# if elapsed > 0:
#     tokens_per_sec = char_len / elapsed
# else:
#     tokens_per_sec = 0

# # 输出结果
# print(f"\n⏱️ 总耗时：{elapsed:.2f} 秒")
# print(f"📝 输出总长度：{char_len} 字符")
# print(f"⚡ 平均生成速度：{tokens_per_sec:.2f} token/s")


# import requests

# url = "https://ragflow.pkubir.cn/v1/kb_api/list"
# params = {
#     "page": 1,
#     "page_size": 10,
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
    "question": "民族复兴",
    "page": 1,
    "size": 10
}

response = requests.post(url, headers=headers, json=data)
result = response.json()

if result["code"] == 0:
    total = result["data"]["total"]
    chunks = result["data"]["chunks"]
    labels = result["data"].get("labels", [])

    print(f"检索到 {total} 个相关chunks")
    print(f"标签: {labels}\n")

    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"  相似度: {chunk.get('similarity', 0):.4f}")
        print(f"  文档: {chunk['docnm_kwd']}")
        print(f"  内容: {chunk['content_with_weight'][:150]}...")
        print()
else:
    print(f"Error: {result.get('message', 'Unknown error')}")