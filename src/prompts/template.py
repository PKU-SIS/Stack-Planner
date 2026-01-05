# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import dataclasses
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.prebuilt.chat_agent_executor import AgentState
from src.config.configuration import Configuration
from src.utils.logger import logger

# Initialize Jinja2 environment
env = Environment(
    loader=FileSystemLoader(os.path.dirname(__file__)),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def get_prompt_template(prompt_name: str, context: dict = None) -> str:
    """
    Load and return a prompt template using Jinja2.

    Args:
        prompt_name: Name of the prompt template file (without .md extension)

    Returns:
        The template string with proper variable substitution syntax
    """
    try:
        template = env.get_template(f"{prompt_name}.md")
        return template.render(**(context or {}))
    except Exception as e:
        raise ValueError(f"Error loading template {prompt_name}: {e}")


def apply_prompt_template(
    prompt_name: str,
    state: AgentState,
    configurable: Configuration = None,
    extra_context: dict = None,
) -> list:
    """
    Apply template variables to a prompt template and return formatted messages.

    Args:
        prompt_name: Name of the prompt template to use
        state: Current agent state containing variables to substitute

    Returns:
        List of messages with the system prompt as the first message
    """
    # Convert state to dict for template rendering
    state_vars = {
        "CURRENT_TIME": datetime.now().strftime("%a %b %d %Y %H:%M:%S %z"),
        **state,
    }
    if "memory_stack" in state_vars and isinstance(state_vars["memory_stack"], str):
        state_vars["memory_stack"] = json.loads(state_vars["memory_stack"])

    # Add configurable variables
    if configurable:
        state_vars.update(dataclasses.asdict(configurable))

    if extra_context:
        state_vars.update(extra_context)

    # logger.debug(f"Applying template {prompt_name} with state: {state_vars}")
    try:
        template = env.get_template(f"{prompt_name}.md")
        system_prompt = template.render(**state_vars)

        # Validate and clean messages - FIX FOR ROLE VALIDATION ERROR
        valid_messages = []
        for msg in state.get("messages", []):
            if isinstance(msg, dict):
                # Skip messages with None role or missing content
                if msg.get("role") and msg.get("content") is not None:
                    valid_messages.append(msg)
                else:
                    logger.warning(
                        f"Skipped invalid dict message: role={msg.get('role')}, has_content={'content' in msg}"
                    )
            elif hasattr(msg, "type") or hasattr(msg, "role"):
                # Handle LangChain message objects (HumanMessage, AIMessage, etc.)
                msg_role = getattr(msg, "type", None) or getattr(msg, "role", None)
                msg_content = getattr(msg, "content", None)

                if msg_role and msg_content is not None:
                    msg_dict = {"role": msg_role, "content": msg_content}
                    # Preserve additional_kwargs if present
                    if hasattr(msg, "additional_kwargs"):
                        msg_dict["additional_kwargs"] = msg.additional_kwargs
                    valid_messages.append(msg_dict)
                else:
                    logger.warning(
                        f"Skipped invalid message object: role={msg_role}, has_content={msg_content is not None}"
                    )

        return [{"role": "system", "content": system_prompt}] + valid_messages
    except Exception as e:
        raise ValueError(f"Error applying template {prompt_name}: {e}")
