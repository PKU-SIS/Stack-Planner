from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import LLMResult
from src.utils.logger import logger
from typing import Any, Dict
import json


class ReactAgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        logger.debug("🧠 [LLM Start] Prompt sent to model:")
        for prompt in prompts:
            logger.debug(prompt)

    def on_llm_end(self, response: LLMResult, **kwargs):
        logger.debug("✅ [LLM End] Model response received:")
        for gen in response.generations:
            logger.debug(gen[0].text)

    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.debug(f"🛠️ [Tool Start] Calling tool: {serialized['name']}")
        logger.debug(f"Input to tool: {input_str}")

    def on_tool_end(self, output, **kwargs):
        logger.debug(f"✅ [Tool End] Tool returned output:")
        logger.debug(output)

    def on_chain_start(self, serialized, inputs, **kwargs):
        logger.debug("⛓️ [Chain Start] Inputs:")
        logger.debug(inputs)

    def on_chain_end(self, outputs, **kwargs):
        logger.debug("🔚 [Chain End] Outputs:")
        logger.debug(outputs)


class ToolResultCallbackHandler(BaseCallbackHandler):
    def __init__(self, agent_instance):
        self.agent = agent_instance  # 保存对 agent 的引用

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        # 保存当前 tool 名称和输入
        self.current_tool = {
            "tool_name": serialized.get("name", "unknown"),
            "output": None,
        }

    def on_tool_end(self, output: Any, **kwargs):
        if self.current_tool:
            self.current_tool["output"] = str(output)
            tool_record_str = json.dumps(self.current_tool, ensure_ascii=False)
            self.agent.tool_results.append(tool_record_str)
            self.current_tool = None
        else:
            # 如果没有 on_tool_start，只记录 output
            tool_record_str = json.dumps(
                {"tool_name": "unknown", "output": str(output)}, ensure_ascii=False
            )
            self.agent.tool_results.append(tool_record_str)
