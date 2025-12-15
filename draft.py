import requests

url = "https://ragflow.pkubir.cn/v1/kb_api/list"
params = {
    "page": 1,
    "page_size": 20,
    "keywords": "",
    "orderby": "create_time",
    "desc": "true"
}
data = {
    "tenant_id": "cbae14fb8c8411f0bf2ecd6543f8a381"  #这里提供的子然账号，XXQG知识库在这上面
}

response = requests.post(url, params=params, json=data)
result = response.json()

if result["code"] == 0:
    kbs = result["data"]["kbs"]
    total = result["data"]["total"]
    print(f"获取到 {total} 个知识库")
    for kb in kbs:
        print(f"- {kb['name']} (ID: {kb['id']})")
else:
    print(f"Error: {result.get('message', 'Unknown error')}")