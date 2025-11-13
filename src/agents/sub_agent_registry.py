from enum import Enum


class SubAgentType(Enum):
    """子Agent类型枚举，定义可委派的专项Agent"""

    RESEARCHER = "researcher"  # 负责信息检索与研究
    CODER = "coder"  # 负责代码生成与执行
    REPORTER = "reporter"  # 负责结果整理与报告生成
    PLANNER = "replanner"  # 负责复杂任务分解和规划
    EXPERIENCE_AGENT = "experience_agent"  # 负责经验总结与提炼


from src.graph.sp_nodes import (
    researcher_node,
    coder_node,
    reporter_node,
    experience_agent_node,
    researcher_xxqg_node,
    reporter_xxqg_node,
    sp_planner_node,
)

# 定义可用的子Agent列表，绑定名称与节点函数
sub_agents_sp = [
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "(Before you try it, use Experience Agent first!!!) Information collection and research, but before you try to research or DELEGATE the RESEARCHER AGENT, invoke the Experience Agent FRIST using only domain-related keywords (without any username) to discover experiences or insights from other users or past cases that may inform your reasoning. And if there are not suffcient results you can decide to research.",
        "node": researcher_node,
    },
    {
        "name": SubAgentType.CODER.value,
        "description": "Code generation and execution for math or code problems",
        "node": coder_node,
    },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
        "node": reporter_node,
    },
    {
        "name": SubAgentType.EXPERIENCE_AGENT.value,
        "description": "The Experience Agent is responsible for searching and retrieving long-term experiences from the centralized memory library. Each memory entry is associated with a username and contains structured long-term knowledge, including core, semantic, and episodic components. You FIRST decision should ALWAYS be query this agent using the username specified to load the user's personal long-term memory context, ensuring that your reasoning starts with all relevant prior knowledge. (!!IMPORTANT!!)  Before you try to research or DELEGATE the RESEARCHER AGENT, invoke the Experience Agent FRIST using only domain-related keywords (without any username) to discover experiences or insights from other users or past cases that may inform your reasoning. And if there are not suffcient results you can decide to research. **AND REMEMBER: you can use it any time if it is necessary for you to query the context or background knowledges (with or without username), especially when you needs information of the user's and your history, intention and role.**",
        "node": experience_agent_node,
    },
]


sub_agents_sp_xxqg = [
    {
        "name": SubAgentType.PLANNER.value,
        # 将问题拆解成方便处理的子任务，来更好的指导任务规划
        "description": "Decompose search problems into manageable subtasks to better guide research step. Don't contain any requirements about report writing in task description, this agent can only handle **search steps planning**. You MUST and Only use it at the beginning of the task.",
        "node": sp_planner_node,
    },
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
        "node": researcher_xxqg_node,
    },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
        "node": reporter_xxqg_node,
    },
]


def get_sub_agents_by_global_type(graph_type: str):
    """
    根据图类型返回可用的子Agent列表
    Args:
        graph_type (str): 图类型，例如 "sp" 或 "sp_xxqg"
    Returns:
        List[Dict]: 包含子Agent名称、节点和描述的列表
    """
    if graph_type == "sp" or graph_type == "base":
        return sub_agents_sp
    elif graph_type == "sp_xxqg":
        return sub_agents_sp_xxqg
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
