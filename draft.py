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

allocations = [
    {"node_id": "node_4", "word_limit": 200},
    {"node_id": "node_5", "word_limit": 180},
    {"node_id": "node_6", "word_limit": 180},
    {"node_id": "node_8", "word_limit": 220},
    {"node_id": "node_9", "word_limit": 200},
    {"node_id": "node_10", "word_limit": 200},
    {"node_id": "node_11", "word_limit": 220},
    {"node_id": "node_13", "word_limit": 160},
    {"node_id": "node_14", "word_limit": 180},
    {"node_id": "node_15", "word_limit": 160},
    {"node_id": "node_18", "word_limit": 180},
    {"node_id": "node_19", "word_limit": 180},
    {"node_id": "node_21", "word_limit": 160},
    {"node_id": "node_22", "word_limit": 160},
    {"node_id": "node_23", "word_limit": 160},
    {"node_id": "node_24", "word_limit": 160},
    {"node_id": "node_26", "word_limit": 180},
    {"node_id": "node_27", "word_limit": 180},
    {"node_id": "node_28", "word_limit": 180},
    {"node_id": "node_29", "word_limit": 180},
    {"node_id": "node_32", "word_limit": 180},
    {"node_id": "node_33", "word_limit": 180},
    {"node_id": "node_34", "word_limit": 180},
    {"node_id": "node_36", "word_limit": 180},
    {"node_id": "node_37", "word_limit": 180},
    {"node_id": "node_38", "word_limit": 180},
    {"node_id": "node_39", "word_limit": 180},
    {"node_id": "node_40", "word_limit": 180},
    {"node_id": "node_42", "word_limit": 180},
    {"node_id": "node_43", "word_limit": 180},
    {"node_id": "node_44", "word_limit": 180},
    {"node_id": "node_45", "word_limit": 180},
    {"node_id": "node_48", "word_limit": 180},
    {"node_id": "node_49", "word_limit": 180},
    {"node_id": "node_50", "word_limit": 180},
    {"node_id": "node_52", "word_limit": 180},
    {"node_id": "node_53", "word_limit": 180},
    {"node_id": "node_54", "word_limit": 180},
    {"node_id": "node_56", "word_limit": 180},
    {"node_id": "node_57", "word_limit": 180},
    {"node_id": "node_58", "word_limit": 180},
    {"node_id": "node_60", "word_limit": 180},
    {"node_id": "node_61", "word_limit": 180},
    {"node_id": "node_62", "word_limit": 180},
    {"node_id": "node_64", "word_limit": 180},
    {"node_id": "node_65", "word_limit": 180},
    {"node_id": "node_66", "word_limit": 180},
    {"node_id": "node_67", "word_limit": 180},
]

total = sum(item["word_limit"] for item in allocations)

print("total_allocated =", total)
