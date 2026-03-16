from typing import List
import json

from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from src.config.configuration import Configuration
from src.agents.CommonReactAgent import CommonReactAgent
from src.utils.logger import logger
from src.llms.mem0 import mem0_search


@tool(description="从 mem0 中批量检索多个问题对应的历史记忆，并返回 JSON 字符串结果。输入必须是字符串列表。")
async def mem0_search_tool(queries: List[str]) -> str:
    try:
        result = await mem0_search(queries, user_id="dev")
        if result is None:
            return "未检索到相关记忆。"
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("mem0_search_tool failed")
        return f"mem0_search 调用失败: {e}"


class Memagent(CommonReactAgent):

    agent_name: str = "memagent"
    description: str = "Memory agent for gathering information and resources from history."

    def __init__(self, *args, **kwargs):
        agent_type = kwargs.pop("agent_type", "default_agent")
        tools = [mem0_search_tool]

        super().__init__(
            agent_name=agent_type,
            tools=tools,
            system_prompt=agent_type,
        )

    async def execute_agent_step(self, state) -> Command:
        observations = state.get("observations", [])

        text = state.get("user_query", {})
        locale = state.get("locale", "zh-CN")

        agent_input = {
            "messages": [
                HumanMessage(
                    content=(
                        f"任务描述\n\n"
                        f"{text}\n\n"
                        f"语言环境\n\n"
                        f"{locale}\n\n"
                        f"调用 mem0_search_tool 进行检索，再根据检索结果生成中文报告。"
                    )
                )
            ]
        }

        result = await self.ainvoke(
            input=agent_input,
        )

        response_content = result["messages"][-1].content
        logger.info(
            f"{self.agent_name.capitalize()} full response: {response_content}"
        )

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name=self.agent_name,
                    )
                ],
                "observations": observations + [response_content],
                # "data_collections": data_collections + self.tool_results,
            },
        )