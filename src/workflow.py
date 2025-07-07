# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import os
from datetime import datetime
from src.utils.logger import logger


def enable_debug_logging():
    """Enable debug level logging."""
    logger.set_log_level(log_level="DEBUG")


# Create the graph
# graph = build_graph_sp()


# NOTE JXK main code直接进入到这里
async def run_agent_workflow_async(
    user_input: str,
    debug: bool = False,
    max_plan_iterations: int = 1,
    max_step_num: int = 3,
    enable_background_investigation: bool = True,
    graph_format: str = "sp",
):
    """Run the agent workflow asynchronously with the given user input.

    Args:
        user_input: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context

    Returns:
        The final state after the workflow completes
    """
    if not user_input:
        raise ValueError("Input could not be empty")

    if debug:
        enable_debug_logging()

    if graph_format == "sp":
        from src.graph.builder import sp_graph as graph
    elif graph_format == "xxqg":
        from src.graph.builder import xxqg_graph as graph
    elif graph_format == "sp_xxqg":
        from src.graph.builder import sp_xxqg_graph as graph

    logger.info(f"Starting async workflow with user input: {user_input}")
    initial_state = {
        # Runtime Variables
        "messages": [{"role": "user", "content": user_input}],
        "auto_accepted_plan": True,
        "enable_background_investigation": enable_background_investigation,
        "user_query": user_input,
    }
    config = {
        "configurable": {
            "thread_id": "default",
            "max_plan_iterations": max_plan_iterations,
            "max_step_num": max_step_num,
            "mcp_settings": {
                "servers": {
                    "mcp-github-trending": {
                        "transport": "stdio",
                        "command": "uvx",
                        "args": ["mcp-github-trending"],
                        "enabled_tools": ["get_github_trending_repositories"],
                        "add_to_agents": ["researcher"],
                    }
                }
            },
        },
        "recursion_limit": 100,
    }
    last_message_cnt = 0
    async for s in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        try:
            if isinstance(s, dict) and "messages" in s:
                if len(s["messages"]) <= last_message_cnt:
                    continue
                last_message_cnt = len(s["messages"])
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    logger.info(message)
                else:
                    message.pretty_logger.info()
            else:
                # For any other output format
                logger.info(f"Output: {s}")
        except Exception as e:
            logger.error(f"Error processing stream output: {e}")
            logger.info(f"Error processing output: {str(e)}")

    logger.info("Async workflow completed successfully")


if __name__ == "__main__":
    logger.info(graph.get_graph(xray=True).draw_mermaid())
