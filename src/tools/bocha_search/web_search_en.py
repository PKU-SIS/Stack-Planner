import os
from src.tools.bocha_search import WebSearcher
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any
from src.utils.logger import logger

class WebSearcherEnglish(WebSearcher):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("BOCHA_API_KEY")
            assert api_key is not None, "LANG_SEARCH_API_KEY is not set"
        self.api_key = api_key
        self.base_url =  "https://api.langsearch.com/v1/web-search" # ←←← 必须有这一行！
        super().__init__(self.api_key,self.base_url)



def web_search(query: str, top_k: int = 10):
    """
    调用 BoCha 英文网络搜索 API，返回结构化搜索结果。
    模仿 search_docs 的风格：带错误处理、日志、结构化输出。
    """
    api_url = "https://api.langsearch.com/v1/web-search"
    api_key = os.getenv("BOCHA_API_KEY")
    if not api_key:
        logger.error("环境变量 BOCHA_API_KEY 未设置")
        return results


    try:
        searcher = WebSearcherEnglish()
        return searcher.search(query, top_k)
    
    except requests.RequestException as e:
        logger.error(f"BoCha 搜索请求异常: {e}")

    return None






if __name__ == "__main__":
    results = web_search("Why the sky is blue?", top_k = 10)
    for result in results:
        print(result)