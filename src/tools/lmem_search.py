from src.utils.logger import logger
from typing import List, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field


from src.config.tools import SELECTED_RAG_PROVIDER
from src.rag import Document, Retriever, Resource, build_retriever, LmemProvider
from src.rag.ragflow import parse_uri


class RetrieverInput(BaseModel):
    keywords: str = Field(description="search keywords to look up")


class RetrieverTool(BaseTool):
    name: str = "local_search_tool"
    description: str = (
        "Useful for retrieving information from the file with `rag://` uri prefix, it should be higher priority than the web search or writing code. Input should be a search keywords."
    )
    args_schema: Type[BaseModel] = RetrieverInput

    retriever: Retriever = Field(default_factory=Retriever)
    resources: list[Resource] = Field(default_factory=list)

    def _run(
        self,
        keywords: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> list[Document]:
        logger.info(f"Retriever tool query: {keywords}")
        documents = self.retriever.query_relevant_documents(keywords, self.resources)
        if not documents:
            return "No results found from the local knowledge base."
        return [doc.to_dict() for doc in documents]

    async def _arun(
        self,
        keywords: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> list[Document]:
        return self._run(keywords, run_manager.get_sync())


def get_lmem_search_tool(resources: List[Resource]) -> RetrieverTool | None:
    if not resources:
        return None
    logger.info(f"create retriever tool")
    retriever = LmemProvider()

    if not retriever:
        return None
    return RetrieverTool(retriever=retriever, resources=resources)


if __name__ == "__main__":
    resources = [
        Resource(
            uri="rag://dataset/39ea834abf1111f0bf2ecd6543f8a381",
            title="LTM",
            description="长期记忆知识库",
        )
    ]
    retriever_tool = get_lmem_search_tool(resources)
    # print(parse_uri("rag://dataset/1c7e2ea4362911f09a41c290d4b6a7f0"))
    # print(retriever_tool.name)
    # print(retriever_tool.description)
    # print(retriever_tool.args)
    print(retriever_tool.invoke("Bardi"))
