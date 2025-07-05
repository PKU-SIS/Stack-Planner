import requests
from langchain_core.tools import tool
from typing import Annotated
from .decorators import log_io
# 8509 学习强国
# 8401 国家安全部知识库
def search_docs(question, top_k = 5):
    docs = []
    api_url = "http://localhost:8509/knowledge_base/search_docs"
    query = {
        "query": question,
        "knowledge_base_name": "学习强国",
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
@log_io
def search_docs_tool(
    question: Annotated[str, "检索的问题，使用语义相似度匹配"],
) -> dict:
    """
    使用这个工具查询本地存储的领域知识库，检索方式为语义相似度匹配，返回与question相关的文档内容。
    """
    docs = search_docs(question, 20)
    #print(docs)
    return {"query": question, "docs": docs}

#todo 知识库的领域分类如何注册到工具调用中？如何根据问题+领域分类，自适应的选择知识库去检索

print(search_docs_tool("习近平总书记关于全面从严治党的重要论述有哪些？"))