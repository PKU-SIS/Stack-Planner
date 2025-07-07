# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .types import State

from src.utils.logger import logger
from src.agents.SubAgentManager import SubAgentManager
from src.agents.CentralAgent import CentralAgent

# -------------------------
# 全局实例与节点定义
# -------------------------
global_central_agent = CentralAgent()
sub_agent_manager = SubAgentManager(global_central_agent)


# 节点处理函数定义
async def central_agent_node(state: State, config: RunnableConfig) -> Command:
    """中枢Agent节点处理函数，触发决策流程"""
    logger.info("中枢Agent节点激活")

    # 执行决策流程
    decision = global_central_agent.make_decision(state, config)
    return global_central_agent.execute_action(decision, state, config)


async def researcher_node(state: State, config: RunnableConfig) -> Command:
    """研究Agent节点处理函数"""
    return await sub_agent_manager.execute_researcher(state, config)


async def coder_node(state: State, config: RunnableConfig) -> Command:
    """编码Agent节点处理函数"""
    return await sub_agent_manager.execute_coder(state, config)


def reporter_node(state: State, config: RunnableConfig) -> Command:
    """报告Agent节点处理函数"""
    return sub_agent_manager.execute_reporter(state, config)

def reporter_xxqg_node(state: State, config: RunnableConfig) -> Command:
    """报告Agent节点处理函数"""
    return sub_agent_manager.execute_xxqg_reporter(state, config)

def researcher_xxqg_node(state: State, config: RunnableConfig) -> Command:
    """研究Agent节点处理函数"""
    return sub_agent_manager.execute_xxqg_researcher(state, config)
