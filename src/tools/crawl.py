from src.utils.logger import logger
from typing import Annotated

from langchain_core.tools import tool
from .decorators import log_io

from src.crawler import Crawler


@tool
@log_io
def crawl_tool(
    url: Annotated[str, "The url to crawl."],
) -> str:
    """Use this to crawl a url and get a readable content in markdown format."""
    try:
        crawler = Crawler()
        article = crawler.crawl(url)
        content = article.to_markdown()
        # 限制输出内容长度，避免淹没控制台
        truncated_content = (
            content[:2000] + "...[内容已截断]" if len(content) > 2000 else content
        )
        return {
            "url": url,
            "crawled_content": truncated_content,
            "full_content_length": len(content),
        }
    except BaseException as e:
        error_msg = f"Failed to crawl. Error: {repr(e)}"
        logger.error(error_msg)
        return error_msg
