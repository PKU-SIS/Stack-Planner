# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT


from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from .types import State

from src.utils.logger import logger
from src.utils.statistics import global_statistics


# -------------------------
# 全局实例与节点定义
# -------------------------
global_central_agent = None
sub_agent_manager = None


def init_agents(graph_format: str):
    from src.agents.SubAgentManager import SubAgentManager
    from src.agents.CentralAgent import CentralAgent

    global global_central_agent, sub_agent_manager
    global_central_agent = CentralAgent(graph_format=graph_format)
    sub_agent_manager = SubAgentManager(global_central_agent)


def _check_agents_initialized():
    if global_central_agent is None or sub_agent_manager is None:
        raise RuntimeError(
            "请先调用 init_agents(graph_format) 初始化 全局的agent 实例。"
        )


# 节点处理函数定义
async def central_agent_node(state: State, config: RunnableConfig) -> Command:
    """中枢Agent节点处理函数，触发决策流程"""
    _check_agents_initialized()
    logger.info("中枢Agent节点激活")

    # 执行决策流程
    decision = global_central_agent.make_decision(state, config)
    return global_central_agent.execute_action(decision, state, config)


async def researcher_node(state: State, config: RunnableConfig) -> Command:
    """研究Agent节点处理函数"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_researcher(state, config)


async def coder_node(state: State, config: RunnableConfig) -> Command:
    """编码Agent节点处理函数"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_coder(state, config)


def reporter_node(state: State, config: RunnableConfig) -> Command:
    """报告Agent节点处理函数"""
    _check_agents_initialized()
    return sub_agent_manager.execute_reporter(state, config)


def reporter_xxqg_node(state: State, config: RunnableConfig) -> Command:
    """报告Agent节点处理函数"""
    _check_agents_initialized()
    return sub_agent_manager.execute_xxqg_reporter(state, config)


async def researcher_xxqg_node(state: State, config: RunnableConfig) -> Command:
    """研究Agent节点处理函数"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_xxqg_researcher(state, config)


async def sp_planner_node(state: State, config: RunnableConfig) -> Command:
    """规划Agent节点处理函数"""
    _check_agents_initialized()
    return sub_agent_manager.execute_sp_planner(state, config)


async def perception_node(state: State, config: RunnableConfig) -> Command:
    """感知层节点处理函数"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_perception(state, config)


async def outline_node(state: State, config: RunnableConfig) -> Command:
    """大纲生成节点处理函数"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_outline(state, config)


async def human_feedback_node(state: State, config: RunnableConfig) -> Command:
    """人工反馈节点处理函数（已废弃，保留用于兼容）"""
    _check_agents_initialized()
    return await sub_agent_manager.execute_human_feedback(state, config)


async def human_agent_node(state: State, config: RunnableConfig) -> Command:
    """
    Human Agent 节点处理函数

    专门负责与人类的交互，包括：
    - form_filling: 表单填写（perception 阶段）
    - outline_confirmation: 大纲确认（outline 阶段）
    - report_feedback: 报告反馈（reporter 阶段）
    - proactive_question: 主动提问（central agent 发起）

    🔴 核心原则：人类反馈优先级最高
    """
    _check_agents_initialized()
    logger.info("Human Agent 节点激活")
    return await sub_agent_manager.execute_human(state, config)
