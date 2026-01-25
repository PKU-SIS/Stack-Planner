from enum import Enum


class SubAgentType(Enum):
    """子Agent类型枚举，定义可委派的专项Agent"""

    RESEARCHER = "researcher"  # 负责信息检索与研究
    CODER = "coder"  # 负责代码生成与执行
    REPORTER = "reporter"  # 负责结果整理与报告生成
    PLANNER = "replanner"  # 负责复杂任务分解和规划
    OUTLINE = "outline"  # 负责大纲生成
    PERCEPTION = "perception"  # 负责表单生成
    HUMAN = "human"  # 负责与人类的交互（表单填写、大纲确认、报告反馈、主动提问）


from src.graph.sp_nodes import (
    researcher_node,
    coder_node,
    reporter_node,
    researcher_xxqg_node,
    reporter_xxqg_node,
    sp_planner_node,
    outline_node,
    perception_node,
    human_feedback_node,
    human_agent_node,
)

# 定义可用的子Agent列表，绑定名称与节点函数
sub_agents_sp = [
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
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
]


sub_agents_sp_xxqg = [
    # {
    #     "name": SubAgentType.PLANNER.value,
    #     # 将问题拆解成方便处理的子任务，来更好的指导任务规划
    #     "description": "Decompose search problems into manageable subtasks to better guide research step. Don't contain any requirements about report writing in task description, this agent can only handle **search steps planning**. You MUST and Only use it at the beginning of the task.",
    #     "node": sp_planner_node,
    # },
    {
        "name": SubAgentType.PERCEPTION.value,
        "description": "Serve as the first sub-agent in the workflow to perform pre-retrieval perception and clarification. This agent identifies missing or ambiguous information before any retrieval or generation, and produces a structured form or questionnaire that is explicitly intended to be returned to the human user for input completion.",
        "node": perception_node,
    },
    # {#这个暂时先不做成子 agent
    #     "name": SubAgentType.HUMAN.value,
    #     "description": "Generate a structured content outline after the overall plan is finalized. This agent designs and adjusts the hierarchical structure of the report, including section titles and logical organization. It does NOT generate full text content or conduct research, and should be used only after task planning is complete.",
    #     "node": human_feedback_node,
    # },
    {
        "name": SubAgentType.OUTLINE.value,
        "description": "Execute after the perception stage and subsequent human feedback to generate a structured task or content outline. This agent uses the original query together with the confirmed user-provided form as inputs to organize and define an outline, and outputs it for human review and confirmation before any central reasoning or content generation begins. It does NOT perform retrieval, reasoning, or full content generation.",
        "node": outline_node,
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
    {
        "name": SubAgentType.HUMAN.value,
        "description": "Handle all human interactions including form filling, outline confirmation, report feedback, and proactive questioning. This agent manages the interrupt mechanism, ensures human feedback is properly collected, and ALWAYS prioritizes human input above all other considerations. 🔴 Human feedback has the HIGHEST priority.",
        "node": human_agent_node,
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
