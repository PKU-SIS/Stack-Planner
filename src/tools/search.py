import json
from src.utils.logger import logger
import os

from langchain_community.tools import BraveSearch, DuckDuckGoSearchResults
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper, BraveSearchWrapper

from src.config import SearchEngine, SELECTED_SEARCH_ENGINE
from src.tools.tavily_search.tavily_search_results_with_images import (
    TavilySearchResultsWithImages,
)
# from src.tools.bocha_search import BoChaSearchResults
from src.tools.bocha_search.bocha_search_sp import BoChaSearchResults
from src.tools.decorators import create_logged_tool
from src.utils.reference_utils import global_reference_map

# Create logged versions of the search tools
LoggedTavilySearch = create_logged_tool(TavilySearchResultsWithImages)
LoggedBoChaSearch = create_logged_tool(BoChaSearchResults)#要加的BoCha
LoggedDuckDuckGoSearch = create_logged_tool(DuckDuckGoSearchResults)
LoggedBraveSearch = create_logged_tool(BraveSearch)
LoggedArxivSearch = create_logged_tool(ArxivQueryRun)


# Get the selected search tool
def get_web_search_tool(max_search_results: int):
    if SELECTED_SEARCH_ENGINE == SearchEngine.TAVILY.value:
        return LoggedTavilySearch(
            name="web_search",
            max_results=max_search_results,
            include_raw_content=False,
            include_images=True,
            include_image_descriptions=True,
        )
    #给BoCha加一个Web搜索、对应的接口实现还没做
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BOCHA.value:
        return LoggedBoChaSearch(name="web_search",max_results=max_search_results)
    elif SELECTED_SEARCH_ENGINE == SearchEngine.DUCKDUCKGO.value:
        return LoggedDuckDuckGoSearch(name="web_search", max_results=max_search_results)
    elif SELECTED_SEARCH_ENGINE == SearchEngine.BRAVE_SEARCH.value:
        return LoggedBraveSearch(
            name="web_search",
            search_wrapper=BraveSearchWrapper(
                api_key=os.getenv("BRAVE_SEARCH_API_KEY", ""),
                search_kwargs={"count": max_search_results},
            ),
        )
    elif SELECTED_SEARCH_ENGINE == SearchEngine.ARXIV.value:
        return LoggedArxivSearch(
            name="web_search",
            api_wrapper=ArxivAPIWrapper(
                top_k_results=max_search_results,
                load_max_docs=max_search_results,
                load_all_available_meta=True,
            ),
        )
    else:
        raise ValueError(f"Unsupported search engine: {SELECTED_SEARCH_ENGINE}")


if __name__ == "__main__":
    # results = LoggedDuckDuckGoSearch(
    #     name="web_search", max_results=3, output_format="list"
    # )
    #到时候看看输出的东西对不对
    tool = LoggedBoChaSearch(
        name="web_search", max_results=3, output_format="list"
    )

    print("工具名称:", tool.name)
    print("工具描述:", tool.description)
    print("工具参数:", tool.args)

    print("\n--- 开始搜索 ---")
    result, raw = tool.invoke({"query": "cute panda"})
    print(json.dumps(result, indent=2, ensure_ascii=False))
