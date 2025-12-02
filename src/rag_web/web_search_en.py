import os
from src.rag_web import WebSearcher

class WebSearcherEnglish(WebSearcher):
    def __init__(self, api_key: str = None):
        if api_key is None:
            api_key = os.getenv("LANG_SEARCH_API_KEY")
            assert api_key is not None, "LANG_SEARCH_API_KEY is not set"

        super().__init__(api_key, "https://api.bocha.cn/v1/web-search")

def web_search(query: str, count: int = 10):
    searcher = WebSearcherEnglish()
    return searcher.search(query, count)

if __name__ == "__main__":
    results = web_search("Why the sky is blue?", count = 10)
    for result in results:
        print(result)