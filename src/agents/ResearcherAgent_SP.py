from src.config.configuration import Configuration
from src.agents.CommonReactAgent import CommonReactAgent
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from src.utils.logger import logger
import os


class ResearcherAgentSP(CommonReactAgent):
    """Agent for conducting research and gathering information."""

    agent_name: str = "researcher"
    description: str = "Researcher agent for gathering information and resources."

    def __init__(self, *args, **kwargs):
        agent_type = kwargs.pop("agent_type", "default_agent")
        print("agent_type",agent_type)
        config = kwargs.pop("config", None)
        default_tools = kwargs.pop("default_tools", []) #default_tools
        """Initialize the ResearcherAgent with additional attributes."""
        configurable = Configuration.from_runnable_config(config)
        mcp_servers = {}
        enabled_tools = {}

        # Extract MCP server configuration for this agent type
        if configurable.mcp_settings:
            for server_name, server_config in configurable.mcp_settings[
                "servers"
            ].items():
                if (
                    server_config["enabled_tools"]
                    and agent_type in server_config["add_to_agents"]
                ):
                    mcp_servers[server_name] = {
                        k: v
                        for k, v in server_config.items()
                        if k in ("transport", "command", "args", "url", "env")
                    }
                    for tool_name in server_config["enabled_tools"]:
                        enabled_tools[tool_name] = server_name

        # Initialize agent with MCP tools or default tools
        if mcp_servers:
            # async with MultiServerMCPClient(mcp_servers) as client:
            #     loaded_tools = default_tools[:]
            #     for tool in client.get_tools():
            #         if tool.name in enabled_tools:
            #             tool.description = (
            #                 f"Powered by '{enabled_tools[tool.name]}'.\n{tool.description}"
            #             )
            #             loaded_tools.append(tool)
            self._initialize_common_agent(agent_type, default_tools)
        else:
            # Use default tools if no MCP servers are configured
            self._initialize_common_agent(agent_type, default_tools)

    def _initialize_common_agent(self, agent_type, tools):
        """Helper method to initialize the CommonReactAgent."""
        super().__init__(agent_name=agent_type, tools=tools, system_prompt=agent_type)

    async def execute_agent_step(self, state) -> Command:
        """Helper function to execute a step using the specified agent."""
        observations = state.get("observations", [])
        data_collections = state.get("data_collections", [])

        # 从 params 中获取任务描述
        params = state.get("delegation_context", {})
        logger.debug(f"Delegate context: {params}")
        task_description = params.get("task_description", "")

        if not task_description:
            logger.warning("No task description found in params")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content="错误: 未找到任务描述",
                            name=self.agent_name,
                        )
                    ],
                    "observations": observations + ["任务描述缺失"],
                },
                goto="research_team",
            )

        logger.info(f"Executing task: {task_description}, agent: {self.agent_name}")

        # 准备 agent 输入，使用任务描述
        agent_input = {
            "messages": [
                HumanMessage(
                    content=f"# Current Task\n\n## Description\n\n{task_description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
                )
            ]
        }

        if state.get("resources"):
            resources_info = "**The user mentioned the following resource files:**\n\n"
            for resource in state.get("resources"):
                resources_info += f"- {resource.title} ({resource.description})\n"

            agent_input["messages"].append(
                HumanMessage(
                    content=resources_info
                    + "\n\n"
                    + "You MUST use the **local_search_tool** to retrieve the information from the resource files.",
                )
            )

        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)"
                "When including images, ONLY use the actual image URLs from the source content. DO NOT create fictional image URLs or placeholders like 'example.com'.",
                name="system",
            )
        )
        # agent_input["messages"].append(
        #     HumanMessage(
        #         content="IMPORTANT: **You have to use search tools** to complete task",
        #         name="system",
        #     )
        # )

        # Invoke the agent
        default_recursion_limit = 25
        try:
            env_value_str = os.getenv(
                "AGENT_RECURSION_LIMIT", str(default_recursion_limit)
            )
            parsed_limit = int(env_value_str)

            if parsed_limit > 0:
                recursion_limit = parsed_limit
                logger.info(f"Recursion limit set to: {recursion_limit}")
            else:
                logger.warning(
                    f"AGENT_RECURSION_LIMIT value '{env_value_str}' (parsed as {parsed_limit}) is not positive. "
                    f"Using default value {default_recursion_limit}."
                )
                recursion_limit = default_recursion_limit
        except ValueError:
            raw_env_value = os.getenv("AGENT_RECURSION_LIMIT")
            logger.warning(
                f"Invalid AGENT_RECURSION_LIMIT value: '{raw_env_value}'. "
                f"Using default value {default_recursion_limit}."
            )
            recursion_limit = default_recursion_limit

        logger.info(f"Agent input: {agent_input}")
        result = await self.ainvoke(
            input=agent_input, config={"recursion_limit": recursion_limit}
        )
        logger.info(
            f"{self.agent_name.capitalize()} result: {result}"
        )
        # Process the result
        response_content = result["messages"][-1].content
        logger.debug(
            f"{self.agent_name.capitalize()} full response: {response_content}"
        )

        logger.info(f"Task execution completed by {self.agent_name}")

        # 后续goto操作在subagentmanager处理
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response_content,
                        name=self.agent_name,
                    )
                ],
                "observations": observations + [response_content],
                "data_collections": data_collections + self.tool_results,
            },
        )
        #     goto="central_agent",
        # )
