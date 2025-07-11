from src.agents.SubAgentConfig import SubAgentType
from ..graph.sp_nodes import (
    researcher_node,
    coder_node,
    reporter_node,
    researcher_xxqg_node,
    reporter_xxqg_node,
)

sub_agents_nodes_sp = {
    SubAgentType.RESEARCHER: researcher_node,
    SubAgentType.CODER: coder_node,
    SubAgentType.REPORTER: reporter_node,
}

sub_agents_nodes_xxqg = {
    SubAgentType.RESEARCHER: researcher_xxqg_node,
    SubAgentType.CODER: coder_node,
    SubAgentType.REPORTER: reporter_xxqg_node,
}
