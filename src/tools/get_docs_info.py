import requests
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from .decorators import log_io
from src.utils.logger import logger
from src.utils.reference_utils import global_reference_map


# 8509 学习强国
# 8401 国家安全部知识库
def search_docs(question, top_k=5):
    docs = []
    api_url = "http://60.28.106.46:8509/knowledge_base/search_docs"
    query = {
        "query": question,
        "knowledge_base_name": "学习强国_new",
        "score_threshold": 1,
        "file_name": "",
        "metadata": {},
        "top_k": top_k,
    }
    try:
        response = requests.post(api_url, json=query)
        if response.status_code == 200:
            results = response.json()
            #results 去重
            seen = set()
            unique_results = []
            for item in results:
                identifier = (item.get("metadata", {}).get("source", ""), item.get("page_content", ""))
                if identifier not in seen:
                    seen.add(identifier)
                    unique_results.append(item)
            results = unique_results

            for result in results:
                metadata = result.get("metadata", "无元数据信息")
                content = result.get("page_content", "无内容信息")
                # if metadata != '无元数据信息':
                #     doc += f"文档来源: {metadata['source']}\n"
                # if content != '无内容信息':
                #     doc += f"切片内容: {content}\n\n"
                if metadata != "无元数据信息" and content != "无内容信息":
                    docs.append(
                        {"source": metadata.get("source", ""), "content": content}
                    )
        else:
            logger.error(
                f"请求失败，状态码: {response.status_code}，错误信息: {response.text}"
            )
        return docs
    except requests.RequestException as e:
        logger.error(f"请求过程中出现异常: {e}")
        return docs

@tool
@log_io
def search_docs_tool(
    question: Annotated[str, "检索的问题，使用语义相似度匹配"],
    config: RunnableConfig
) -> dict:
    """
    使用这个工具查询本地存储的领域知识库，检索方式为语义相似度匹配，返回与question相关的文档内容。
    """
    logger.debug(f"config:{config}")
    session_id = config["configurable"]["thread_id"]
    docs = search_docs(question, 20)
    # return {"query": question, "docs": docs}

    if session_id is None:
        logger.error("session_id is None in config")
        return {"query": question, "docs": docs}
    #logger.debug(f"检索到的文档{docs}")
    ids = global_reference_map.add_references(session_id, docs)
    if not ids or not docs:
        logger.warning("ids or docs is empty, returning empty result")
        return {"query": question, "docs": []}
    # 先把docs按ids升序排序
    # ["【文档x】name\ncontent\n",...]
    ids , docs = zip(*sorted(zip(ids, docs)))
    # rename_docs = ["【文档" + str(doc_id) + "】" + doc.get("source", "") + "\n" + doc.get("content", "") for doc_id, doc in zip(ids, docs)]
    rename_docs = [{"source":"【文档" + str(doc_id) + "】" + doc.get("source", ""), "content": doc.get("content", "")} for doc_id, doc in zip(ids, docs)]
    return {"query": question, "docs": rename_docs}


# todo 知识库的领域分类如何注册到工具调用中？如何根据问题+领域分类，自适应的选择知识库去检索

# logger.info(search_docs_tool("习近平总书记关于全面从严治党的重要论述有哪些？"))
