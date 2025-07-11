from enum import Enum


class SubAgentType(Enum):
    """子Agent类型枚举，定义可委派的专项Agent"""

    RESEARCHER = "researcher"  # 负责信息检索与研究
    CODER = "coder"  # 负责代码生成与执行
    REPORTER = "reporter"  # 负责结果整理与报告生成


# 定义可用的子Agent列表，绑定名称与节点函数
sub_agents_sp = [
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
    },
    {
        "name": SubAgentType.CODER.value,
        "description": "Code generation and execution for math or code problems",
    },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
    },
]


sub_agents_xxqg = [
    {
        "name": SubAgentType.RESEARCHER.value,
        "description": "Information collection and research",
    },
    {
        "name": SubAgentType.CODER.value,
        "description": "Code generation and execution for math or code problems",
    },
    {
        "name": SubAgentType.REPORTER.value,
        "description": "Result organization and report generation",
    },
]


def get_sub_agents_by_global_type(graph_type: str):
    """
    根据图类型返回可用的子Agent列表
    Args:
        graph_type (str): 图类型，例如 "sp" 或 "xxqg"
    Returns:
        List[Dict]: 包含子Agent名称、节点和描述的列表
    """
    if graph_type == "sp":
        return sub_agents_sp
    elif graph_type == "xxqg":
        return sub_agents_xxqg
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
