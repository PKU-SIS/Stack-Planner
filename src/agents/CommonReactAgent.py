

from datetime import datetime
import time
import asyncio
from pydantic import BaseModel,Field, model_validator
from langchain_core.messages import AnyMessage
from langchain.tools import BaseTool
from langgraph.types import Command
from abc import ABCMeta, abstractmethod
from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.llms.llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP

from typing import (
    Optional,
    List
)


class CommonReactAgent(BaseModel, metaclass=ABCMeta):

    agent_name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")
    system_prompt: Optional[str] = Field(
        None, description="System instruction prompt"
    )
    model_config_name: str = Field(default="gpt-4o-mini", description="model wrapper name")
    tools: List[BaseTool] = Field(default=[])

    tool_names: List[str] = Field(default=[])

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination")
    current_step: int = Field(default=0, description="Current step in execution")


    def model_post_init(self, __context) -> None:
        self._agent = create_react_agent(
            name=self.agent_name,
            model=get_llm_by_type(AGENT_LLM_MAP[self.agent_name]),
            tools=self.tools,
            prompt=lambda state: apply_prompt_template(self.system_prompt, state),
        )
    
    async def ainvoke(self, *args, **kwargs):
        return await self._agent.ainvoke(*args, **kwargs)

    @abstractmethod
    def execute_agent_step(self, state) -> Command:
        """
        Execute a single step of the agent's logic.
        This method should be implemented by subclasses to define the agent's behavior.
        """