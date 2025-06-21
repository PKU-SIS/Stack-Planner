from src.config.configuration import Configuration
from src.agents.CommonReactAgent import CommonReactAgent
from langgraph.types import Command
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from src.utils.logger import logger
import os


class ResearcherAgent(CommonReactAgent):
    """Agent for conducting research and gathering information."""

    agent_name: str = "researcher"
    description: str = "Researcher agent for gathering information and resources."

    def __init__(self, *args, **kwargs):
        agent_type = kwargs.pop("agent_type", "default_agent")
        config = kwargs.pop("config", None)
        default_tools = kwargs.pop("default_tools", [])
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
        current_plan = state.get("current_plan")
        observations = state.get("observations", [])

        # Find the first unexecuted step
        current_step = None
        completed_steps = []
        for step in current_plan.steps:
            if not step.execution_res:
                current_step = step
                break
            else:
                completed_steps.append(step)

        if not current_step:
            logger.warning("No unexecuted step found")
            return Command(goto="research_team")

        logger.info(f"Executing step: {current_step.title}, agent: {self.agent_name}")

        # Format completed steps information
        completed_steps_info = ""
        if completed_steps:
            completed_steps_info = "# Existing Research Findings\n\n"
            for i, step in enumerate(completed_steps):
                completed_steps_info += f"## Existing Finding {i + 1}: {step.title}\n\n"
                completed_steps_info += (
                    f"<finding>\n{step.execution_res}\n</finding>\n\n"
                )

        # Prepare the input for the agent with completed steps info
        agent_input = {
            "messages": [
                HumanMessage(
                    content=f"{completed_steps_info}# Current Task\n\n## Title\n\n{current_step.title}\n\n## Description\n\n{current_step.description}\n\n## Locale\n\n{state.get('locale', 'en-US')}"
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
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

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

        # Process the result
        response_content = result["messages"][-1].content
        logger.debug(
            f"{self.agent_name.capitalize()} full response: {response_content}"
        )

        # Update the step with the execution result
        current_step.execution_res = response_content
        logger.info(
            f"Step '{current_step.title}' execution completed by {self.agent_name}"
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
            },
            goto="research_team",
        )
