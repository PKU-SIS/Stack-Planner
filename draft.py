import requests



def get_kb_id_by_name(kb_name):
    url = "https://ragflow.pkubir.cn/v1/kb_api/list"
    params = {
        "page": 1,
        "page_size": 100,
        "keywords": kb_name,
        "orderby": "create_time",
        "desc": "true",
    }
    data = {
        "tenant_id": "e38fafc3e07411f0bf2ecd6543f8a381",#"cbae14fb8c8411f0bf2ecd6543f8a381"  #这里提供的子然账号，XXQG知识库在这上面
        "owner_ids": ["cbae14fb8c8411f0bf2ecd6543f8a381", "dc55bde9b62911f0bf2ecd6543f8a381"]
    }

    try:
        resp = requests.post(url, params=params, json=data, timeout=10)
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            logger.error(f"获取知识库列表失败: {result}")
            return None

        for kb in result.get("data", {}).get("kbs", []):
            if kb.get("name") == kb_name:
                return kb.get("id")

        logger.warning(f"未找到知识库: {kb_name}")
        return None

    except requests.RequestException as e:
        logger.error(f"请求知识库列表接口异常: {e}")
        return None
    except Exception as e:
        logger.error(f"解析知识库列表异常: {e}")
        return None




# 新接口
def search_docs(question, top_k=5):
    docs = []
    knowledge_base_name="学习强国"
    kb_id = get_kb_id_by_name(knowledge_base_name)
    api_url = "https://ragflow.pkubir.cn/v1/chunk_api/retrieval_test"

    query = {
        "tenant_id": "e38fafc3e07411f0bf2ecd6543f8a381",  # "cbae14fb8c8411f0bf2ecd6543f8a381",
        "owner_ids": ["cbae14fb8c8411f0bf2ecd6543f8a381", "dc55bde9b62911f0bf2ecd6543f8a381"],
        "kb_id": [kb_id],
        "similarity_threshold": 0.3,  # 相似度阈值
        "question": question,
        "page": 1,
        "size": top_k,
    }
    try:
        response = requests.post(api_url, json=query)
        # logger.info(f"response: {response}")
        if response.status_code == 200:
            results = response.json()
            # logger.info(f"results: {results}")

            # results 去重
            chunks = results.get("data", {}).get("chunks", [])
            seen = set()
            for chunk in chunks:
                source = chunk.get("docnm_kwd", "")
                content = chunk.get("content_with_weight", "")

                if not source or not content:
                    continue

                # 去重依据：文档名 + 内容
                identifier = (source, content)
                if identifier in seen:
                    continue
                seen.add(identifier)

                docs.append({"source": source, "content": content})
        else:
            logger.error(
                f"请求失败，状态码: {response.status_code}，错误信息: {response.text}"
            )
        return docs
    except requests.RequestException as e:
        logger.error(f"请求过程中出现异常: {e}")
        return docs

print(search_docs("湘北县扶贫", top_k=15))