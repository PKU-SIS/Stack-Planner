import requests
from langchain_core.tools import tool

def search_docs(question, top_k = 5):
    docs = []
    api_url = "http://60.28.106.46:8551/knowledge_base/search_docs"
    query = {
        "query": question,
        "knowledge_base_name": "协和知识库",
        "score_threshold": 1,
        "file_name": "",
        "metadata": {},
        "top_k":top_k
    }
    try:
        response = requests.post(api_url, json=query)
        if response.status_code == 200:
            results = response.json()
            for result in results:
                metadata = result.get('metadata', '无元数据信息')
                content = result.get('page_content', '无内容信息')
                # if metadata != '无元数据信息':
                #     doc += f"文档来源: {metadata['source']}\n"
                # if content != '无内容信息':
                #     doc += f"切片内容: {content}\n\n"
                if metadata != '无元数据信息' and content != '无内容信息':
                    docs.append({
                        "source": metadata.get('source', ''),
                        "content": content
                    })
        else:
            print(f"请求失败，状态码: {response.status_code}，错误信息: {response.text}")
        return docs
    except requests.RequestException as e:
        print(f"请求过程中出现异常: {e}")
        return docs


@tool
def search_docs_tool(
    question: str,
) -> dict:
    """
    查询知识库，返回相关文档内容。
    """
    docs = search_docs(question, 10)
    print(docs)
    return {"query": question, "docs": docs}