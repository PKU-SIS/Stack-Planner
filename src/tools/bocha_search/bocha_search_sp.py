import json
from typing import Dict, List, Optional, Tuple, Union
from langchain_core.runnables import RunnableConfig
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool

from src.utils.reference_utils import global_reference_map

from src.tools.bocha_search.web_search_en import WebSearcherEnglish   # 👈 正确

from src.utils.logger import logger


class BoChaSearchResults(BaseTool):
    """Tool that performs web search using BoCha Search API.

    Similar interface & behavior to TavilySearchResultsWithImages.
    """

    name: str = "web_search"
    description: str = (
        "Use BoCha Search to search the web. "
        "Useful for retrieving recent or general information from the internet."
    )

    # parameters (matching Tavily style)
    max_results: int = 5

    # You can extend these later if BoCha supports more features
    include_raw_content: bool = False

    # Injected searcher (we create it inside _run)
    _searcher: Optional[WebSearcherEnglish] = None

    def _get_searcher(self) -> WebSearcherEnglish:
        """Lazy initialization of the BoCha searcher."""
        if self._searcher is None:
            self._searcher = WebSearcherEnglish()
        return self._searcher

    # ---------------------- Sync Search ----------------------

    def _run(
        self,
        query: str,
        config: RunnableConfig,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict], str], Dict]:
        """Use the BoCha search tool synchronously."""
        logger.debug(f"config:{config}")
        session_id = config["configurable"]["thread_id"]
        
        
        try:
            searcher = self._get_searcher()
            raw_results = searcher.search(query, self.max_results)

        except Exception as e:
            logger.error(f"BoChaSearch error: {e}")
            return repr(e), {}
        # Convert results to unified format (similar to Tavily)
        cleaned_results = self._clean_results(raw_results)

        logger.info(f"BoCha搜索完成: 找到 {len(cleaned_results)} 个结果")
        
        
        if session_id is None:
            logger.error("session_id is None in config")
            return cleaned_results, {"raw_results": raw_results}
        ids = global_reference_map.add_references(session_id,cleaned_results)
        if not ids or not cleaned_results:
            logger.warning("ids or cleaned_results is empty, returning empty result")
            return cleaned_results, {"raw_results": raw_results}

        # 先把cleaned_results按ids升序排序
        # ["【文档x】name\ncontent\n",...]
        ids , cleaned_results = zip(*sorted(zip(ids, cleaned_results)))
        rename_docs = [{"url":"【链接" + str(doc_id) + "】" + doc.get("url", ""), "content": doc.get("content", "")} for doc_id, doc in zip(ids, cleaned_results)]    
            
        return rename_docs, {"raw_results": raw_results}

    # ---------------------- Async Search ----------------------

    async def _arun(
        self,
        query: str,
        config: RunnableConfig,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> Tuple[Union[List[Dict], str], Dict]:
        """Use the BoCha search tool asynchronously."""
        
        logger.debug(f"config:{config}")
        session_id = config["configurable"]["thread_id"]        
        
        try:
            searcher = self._get_searcher()
            raw_results = await searcher.search_async(query, self.max_results)

        except Exception as e:
            logger.error(f"BoChaSearch async error: {e}")
            return repr(e), {}

        cleaned_results = self._clean_results(raw_results)

        logger.info(f"BoCha异步搜索完成: 找到 {len(cleaned_results)} 个结果")

        if session_id is None:
            logger.error("session_id is None in config")
            return cleaned_results, {"raw_results": raw_results}
        ids = global_reference_map.add_references(session_id,cleaned_results)
        if not ids or not cleaned_results:
            logger.warning("ids or cleaned_results is empty, returning empty result")
            return cleaned_results, {"raw_results": raw_results}

        # 先把cleaned_results按ids升序排序
        # ["【文档x】name\ncontent\n",...]
        ids , cleaned_results = zip(*sorted(zip(ids, cleaned_results)))
        rename_docs = [{"url":"【链接" + str(doc_id) + "】" + doc.get("url", ""), "content": doc.get("content", "")} for doc_id, doc in zip(ids, cleaned_results)]    
            
        return rename_docs, {"raw_results": raw_results}

    # ---------------------- Result Processing ----------------------

    def _clean_results(self, raw_results: List[Dict]) -> List[Dict]:
        """Convert BoCha results into a unified structure."""
        cleaned = []
        for item in raw_results:
            cleaned_item = {
                "type": "page",
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "content": item.get("snippet", item.get("content", "")),
            }
            cleaned.append(cleaned_item)

        return cleaned
