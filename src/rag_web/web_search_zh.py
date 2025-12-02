import os
from src.rag_web import WebSearcher

class WebSearcherChinese(WebSearcher):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("BOCHA_API_KEY")
            assert api_key is not None, "BOCHA_API_KEY is not set"

        super().__init__(api_key, "https://api.bocha.cn/v1/web-search")

def web_search(query: str, count: int = 10):
    searcher = WebSearcherChinese()
    return searcher.search(query, count)

if __name__ == "__main__":
    results = web_search("酶是蛋白质吗？", count = 8)
    for result in results:
        print(result)